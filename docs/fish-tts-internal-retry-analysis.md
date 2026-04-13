# Fish TTS Internal Retry — Pitfall Analysis

## Proposed Architecture

On Fish `finish reason=error`:
1. **Don't close the audio context** — keep `_handle_audio_context` draining the queue
2. **Don't push `TTSErrorFrame`** — bypass the external retry strategy entirely
3. Reconnect WS, resend the failed text (from `_tts_text_by_context`)
4. Subsequent `run_tts` calls (S3, S4...) will naturally use the new WS
5. After 3 internal failures → push `TTSErrorFrame`, close context, let external failover handle it

## Key Assumption

**Fish errors are all-or-nothing.** When S2 fails, it returns zero audio — no partial audio
duplication concern.

## How Sentence Flow Actually Works

LLM streams tokens → text aggregator buffers → sentence boundary detected → `_push_tts_frames(S_n)` called → `run_tts` sends `text+flush` to Fish WS → **returns immediately** (`yield None`) → next token processed.

**All sentences within a turn are sent to Fish in quick succession on the same WS**, without waiting for audio to come back. Audio arrives asynchronously via `_receive_messages` and is appended to the shared audio context queue.

Frame processing pauses only at `LLMFullResponseEndFrame` (end of turn) or `BotStoppedSpeakingFrame` — **not between sentences**.

## What Happens When S2 Errors

```
Timeline:
T0: run_tts(S1) → text+flush sent to Fish WS
T1: run_tts(S2) → text+flush sent to Fish WS
T2: run_tts(S3) → text+flush sent to Fish WS
T3: Fish returns audio(S1) → appended to ctx-A queue → played ✓
T4: Fish returns finish(ok) for S1
T5: Fish returns finish(error) for S2 — zero audio returned
    → S3 was already sent on this WS but Fish won't process it
```

At T5, S3's text+flush is "in flight" on a WS that is about to be reconnected.

## Pitfalls

### **1. S3 is already sent on the dead WS — it will be silently lost**

This is the biggest issue. By T5, `run_tts(S3)` has already executed and returned.
The text was sent to Fish via the old WS. After reconnect, Fish has no memory of S3.

**However:** `_tts_text_by_context` only stores the **latest** text per context_id
(line 599: `self._tts_text_by_context[context_id] = text`). Since all sentences share one
context_id, only S3's text is stored — S2's text is overwritten.

**Fix required:** Change `_tts_text_by_context` to store a **list** of texts per context,
or add a separate `_pending_texts` list that tracks all sent-but-unfinished texts.
On reconnect, resend all of them (S2 and S3) in order.

### **2. Fish may return `finish(error)` for S2 but `finish(ok)` for S3 — or vice versa**

We're assuming Fish processes text+flush events sequentially and an error on S2 kills the
entire WS session. **If Fish can error on S2 but still successfully process S3**, then:
- We reconnect and resend S2 → S2 audio arrives after S3 audio → out of order

**Mitigation:** Verify Fish behavior. If Fish errors kill the session (connection closes or
all subsequent requests fail), this is not a concern. The `_receive_messages` loop would
exit on WS close anyway.

### **3. `_receive_messages` returns after scheduling reconnect — new receive loop timing**

Current flow:
```python
self._schedule_reconnect_after_error()
return  # exits _receive_messages
```

After reconnect, a new `_receive_task` is created with a fresh `_receive_messages`.
The old receive loop is gone. The new one processes audio from the new WS.

**Potential issue:** Between the old receive loop exiting and the new one starting,
if `_handle_audio_context` times out (3s), the context is cleaned up.

**Fix:** Send keepalives to the audio context during reconnect. The existing
`_refresh_audio_context(context_id)` method does exactly this — it puts a
`_CONTEXT_KEEPALIVE` sentinel in the queue to reset the 3s timeout.

### **4. `get_active_audio_context_id()` returns `_playing_context_id` — wrong context for resent audio?**

After reconnect, `_receive_messages` calls `get_active_audio_context_id()` when audio
arrives. If ctx-A is still being drained by `_handle_audio_context`, then
`_playing_context_id == ctx-A` → audio goes to the right place. ✓

**But:** If ctx-A timed out during reconnect (pitfall #3), `_playing_context_id` is reset.
The resent audio has no context to land in. It would be silently dropped by
`append_to_audio_context` (logged as "unable to append").

**Fix:** Pitfall #3's fix (keepalives) prevents this.

### **5. Reconnect + resend is not atomic with `run_tts`**

`run_tts` calls happen from `process_frame` → `_push_tts_frames`. The reconnect happens
in `_reconnect_after_error` (a background task). These are concurrent.

**Scenario:**
```
T5: Fish error on S2 → schedule reconnect
T6: _reconnect_after_error starts: disconnect WS
T7: Meanwhile, LLM emits S4 → _push_tts_frames → run_tts(S4) 
    → tries to send on WS → WS is disconnected → exception
```

S4's `run_tts` will hit the `except` block (line 648) which yields `ErrorFrame` + 
`TTSStoppedFrame` and calls `_disconnect` + `_connect`. This races with the ongoing
`_reconnect_after_error`.

**Fix:** During internal retry, set a flag that makes `run_tts` wait for reconnect
to complete before sending. Or use a lock/event to serialize WS access during reconnect.

### **6. `_tts_text_by_context` cleanup**

Currently, error handler pops text from `_tts_text_by_context` (line 530). During internal
retry, we must **not** pop it. But the text is overwritten per-sentence anyway (pitfall #1).

If we switch to a list-based approach, cleanup becomes: clear the list after all retries
succeed or after exhaustion.

### **7. The `finish(ok)` path also pops from `_tts_text_by_context`**

Line 551: `self._tts_text_by_context.pop(context_id, None)` on `finish(ok)`.
If we store multiple texts and S1 finishes OK, it would pop the entire entry —
losing S2 and S3's texts.

**Fix:** Use a per-context ordered list and pop only the first entry on each `finish(ok)`.

## Severity Summary

| # | Pitfall | Severity | Fix Complexity |
|---|---------|----------|----------------|
| 1 | S3 text lost (overwritten in dict) | **Critical** | Medium — change to ordered list |
| 2 | Fish partial success on same WS | **Low** | Verify Fish behavior |
| 3 | Context timeout during reconnect | **High** | Low — add keepalive calls |
| 4 | Wrong context for resent audio | **High** | Solved by #3 |
| 5 | Race between reconnect and run_tts | **Medium** | Medium — add synchronization |
| 6 | Text cleanup during retry | **Low** | Follows from #1 fix |
| 7 | finish(ok) pops shared text entry | **High** | Follows from #1 fix |

## Recommended Implementation Order

1. Change `_tts_text_by_context` from `dict[str, str]` to `dict[str, list[str]]` (ordered queue of pending texts per context)
2. On `finish(ok)`, pop only the first text from the list
3. On `finish(error)`, don't close context, add keepalive, schedule reconnect
4. After reconnect, resend all remaining texts in the list
5. Add a reconnect-in-progress flag/event to make `run_tts` wait during reconnect
6. After 3 failures on the same text, push `TTSErrorFrame` and close context
