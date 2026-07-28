#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ordered sequencer for AggregatedTextFrame slots through TTS processing."""

from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    Frame,
    TTSTextFrame,
)
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker


@dataclass
class _AggregatedFrameSlot:
    """Ordered slot tracking one AggregatedTextFrame through TTS processing.

    Every frame that passes through _push_tts_frames — whether spoken or skipped —
    occupies a slot in the sequencer. Skipped frames wait at their position and are
    emitted downstream only after all preceding spoken slots are complete, preserving
    correct context ordering.
    """

    frame: AggregatedTextFrame
    context_id: str
    spoken: bool
    tracker: WordCompletionTracker | None = None
    transport_destination: str | None = None
    complete: bool = False
    includes_inter_frame_spaces: bool = False


class AggregatedFrameSequencer:
    """Sequences AggregatedTextFrame slots to preserve TTS context ordering.

    Manages an ordered queue of spoken and skipped TTS slots. Spoken slots are tracked
    via a :class:`WordCompletionTracker`; skipped slots (e.g. code blocks excluded from
    TTS synthesis) wait in-place until all preceding spoken slots are complete, then are
    flushed downstream with ``append_to_context=True``.

    All methods are synchronous and return lists of frames the caller should push
    downstream, making the sequencer fully testable without any async machinery.

    Example::

        sequencer = AggregatedFrameSequencer()
        sequencer.register_spoken(frame, ctx_id, tracker, append_to_context=True)
        for f in sequencer.process_word("hello", pts=1000, context_id=ctx_id):
            await self.push_frame(f)
    """

    def __init__(self, name: str = "AggregatedFrameSequencer"):
        """Initialize the sequencer.

        Args:
            name: Label used in log messages (typically the owning TTS service name).
        """
        self._name = name
        self._slots: list[_AggregatedFrameSlot] = []
        self._context_append_to_context: dict[str, bool] = {}

    def register_spoken(
        self,
        frame: AggregatedTextFrame,
        context_id: str,
        tracker: WordCompletionTracker | None,
        append_to_context: bool,
        includes_inter_frame_spaces: bool = False,
    ) -> None:
        """Register a spoken AggregatedTextFrame slot.

        Called from _push_tts_frames for frames sent to the TTS service. The slot is
        marked complete either via :meth:`process_word` (word-timestamp services) or
        :meth:`complete_spoken_slot` (push_text_frames=True services).

        Args:
            frame: The AggregatedTextFrame being spoken.
            context_id: The TTS context ID assigned to this frame.
            tracker: WordCompletionTracker for word-timestamp services; None for
                push_text_frames=True services (they complete via complete_spoken_slot).
            append_to_context: Whether word frames built for this context should carry
                append_to_context=True.
            includes_inter_frame_spaces: When True, every TTSTextFrame emitted for this
                slot carries ``includes_inter_frame_spaces=True`` so downstream consumers
                do not inject extra spaces between consecutive frames.
        """
        self._context_append_to_context[context_id] = append_to_context
        self._slots.append(
            _AggregatedFrameSlot(
                frame=frame,
                context_id=context_id,
                spoken=True,
                tracker=tracker,
                includes_inter_frame_spaces=includes_inter_frame_spaces,
            )
        )

    def register_skipped(
        self,
        frame: AggregatedTextFrame,
        context_id: str,
        transport_destination: str | None,
    ) -> list[Frame]:
        """Register a skipped AggregatedTextFrame and attempt an immediate flush.

        The frame is appended as a skipped slot. If no incomplete spoken slot precedes
        it, the frame is returned right away; otherwise it waits until a later
        :meth:`flush` unblocks it.

        Args:
            frame: The skipped AggregatedTextFrame (e.g. a code block).
            context_id: The context ID assigned in _push_tts_frames.
            transport_destination: Transport routing value to attach at flush time.

        Returns:
            Frames to push downstream (empty when blocked by a preceding spoken slot).
        """
        frame.context_id = context_id
        self._slots.append(
            _AggregatedFrameSlot(
                frame=frame,
                context_id=context_id,
                spoken=False,
                transport_destination=transport_destination,
            )
        )
        return self.flush()

    def process_word(
        self,
        word: str,
        pts: int,
        context_id: str | None,
        includes_inter_frame_spaces: bool = False,
    ) -> list[Frame]:
        """Process one word-timestamp event and return frames to push downstream.

        Locates the active (first incomplete spoken) slot with a tracker, advances it
        by the incoming word, and builds a :class:`TTSTextFrame`. Handles:

        - Words from a context that was never registered or was wiped by
          :meth:`clear` on interruption: dropped as stale (returns an empty list).
        - Normal words that fit entirely within the active slot.
        - Overflow words straddling two slot boundaries.
        - Force-complete when the TTS drops an event (word belongs to the next slot).
        - Passthrough for words not recognised by any slot.
        - Symbol-only words whose llm consumption was skipped (punctuation already
          carried by an adjacent word's span) are emitted with
          ``append_to_context=False`` to avoid duplicating punctuation in context.
        - Flushes any skipped slots unblocked by slot completion.

        Args:
            word: A word token from the TTS service word-timestamp stream.
            pts: Presentation timestamp (nanoseconds) to assign to the frame.
            context_id: TTS context ID from the word-timestamp event.
            includes_inter_frame_spaces: Stamped onto the emitted TTSTextFrame so
                downstream consumers know not to inject extra spaces between frames.

        Returns:
            Ordered list of frames (TTSTextFrame and/or AggregatedTextFrame) to push.
        """
        # Drop words from contexts we never registered or that were wiped by clear()
        # on interruption. Such a word is stale (e.g. delayed word-timestamps the TTS
        # server delivers seconds after the context was cancelled); emitting it would
        # interleave it into the current turn's transcript. A None context_id is left
        # untouched: services without audio contexts legitimately use the passthrough
        # path below.
        if context_id is not None and context_id not in self._context_append_to_context:
            logger.debug(
                f"{self._name} Dropping stale word '{word}' from unknown/cleared "
                f"context {context_id}"
            )
            return []

        active = self._get_active_slot(context_id)
        is_complete = False
        raw_overflow_word = None

        # Routing straight to a matching context can skip over older incomplete
        # slots. Those slots' audio is already done (their context only stays
        # open because the provider stopped sending timestamps for it), so retire
        # them now. The queue-order path reached the same end state via
        # force-complete; doing it explicitly keeps skipped frames behind them
        # from being blocked forever.
        pending_flush: list[Frame] = []
        if active is not None and active.context_id == context_id:
            pending_flush = self._retire_slots_before(active)

        # A symbol-only word (normalizes to nothing — punctuation, emoji) that no
        # slot claims is most commonly a trailing-punctuation timestamp event
        # arriving *after* its slot already completed (completion is driven by
        # alnum counts, so e.g. Cartesia's sentence-final "。" lands here every
        # sentence). The slot's llm_text consumption already included that
        # punctuation, so appending the passthrough to the context would
        # duplicate it ("…ですか？？"). Emit it for word-timestamp consumers but
        # keep it out of the context.
        is_symbol_only = not WordCompletionTracker._normalize(word)

        if active is None and is_symbol_only:
            logger.debug(
                f"{self._name} Symbol word '{word}' arrived with no active slot, "
                "emitting as non-context passthrough"
            )
            frame = self._build_word_frame(
                word,
                pts,
                context_id,
                includes_inter_frame_spaces=includes_inter_frame_spaces,
            )
            frame.append_to_context = False
            return [frame]

        if active and active.tracker:
            if not active.tracker.word_belongs_here(word):
                next_slot = self._get_next_active_slot(active)
                word_fits_next = (
                    next_slot is not None
                    and next_slot.tracker is not None
                    and next_slot.tracker.word_belongs_here(word)
                )
                if not word_fits_next:
                    log = logger.debug if is_symbol_only else logger.warning
                    log(
                        f"{self._name} Word '{word}' not recognised by any slot, "
                        "emitting as passthrough"
                    )
                    frame = self._build_word_frame(
                        word,
                        pts,
                        context_id,
                        includes_inter_frame_spaces=includes_inter_frame_spaces,
                    )
                    if is_symbol_only:
                        frame.append_to_context = False
                    return [*pending_flush, frame]

            is_complete = active.tracker.add_word_and_check_complete(word)
            raw_overflow_word = active.tracker.get_overflow_word()

        # Give preference to the per-call flag; fall back to the slot's flag.
        # Also propagate the per-call flag onto the slot so force_complete inherits it.
        if active and includes_inter_frame_spaces:
            active.includes_inter_frame_spaces = True
        slot_ifs = includes_inter_frame_spaces or (
            active.includes_inter_frame_spaces if active else False
        )

        frame_text = (
            active.tracker.get_word_for_frame() if (active and active.tracker) else word
        ) or word
        raw_text = active.tracker.get_llm_consumed() if (active and active.tracker) else None
        emit_context_id = active.context_id if active else context_id

        # logger.debug(f"{self._name} Word '{word}' → frame_text='{frame_text}', raw='{raw_text}'")
        word_frame = self._build_word_frame(
            frame_text,
            pts,
            emit_context_id,
            raw_text=raw_text,
            includes_inter_frame_spaces=slot_ifs,
        )

        # A symbol-only token the slot claimed (it sits in the cursor window) but
        # whose llm consumption was skipped is punctuation already carried by an
        # adjacent word's span — e.g. Cartesia ja sends a mid-frame "、" as its
        # own timestamp event after the preceding chunk's sweep consumed it.
        # Appending it again would duplicate punctuation in the context
        # ("ご用件、、"), so emit it for word-timestamp consumers only.
        if (
            is_symbol_only
            and active
            and active.tracker
            and active.tracker.last_symbol_consumption_skipped
        ):
            word_frame.append_to_context = False

        # Frames released by retiring older slots precede this word in the
        # timeline, so they must be emitted first.
        frames: list[Frame] = [*pending_flush, word_frame]

        if active and active.tracker:
            frames.append(self._build_progress_frame(active, pts))

        if is_complete and active:
            active.complete = True
            frames.extend(self.flush(last_word_pts=pts))
            if raw_overflow_word:
                logger.debug(f"{self._name} Emitting overflow word '{raw_overflow_word}'")
                frames.extend(self.process_word(raw_overflow_word, pts, context_id))

        return frames

    def complete_spoken_slot(self) -> list[Frame]:
        """Mark the first pending spoken slot complete and flush unblocked skipped frames.

        Used by push_text_frames=True services: after the TTSTextFrame has been appended
        to the audio context, this marks the spoken slot done and releases any skipped
        frames waiting behind it.

        Returns:
            AggregatedTextFrame(s) that are now unblocked and should be pushed.
        """
        slot = next((s for s in self._slots if s.spoken and not s.complete), None)
        if slot:
            slot.complete = True
        return self.flush()

    def flush(self, last_word_pts: int | None = None) -> list[Frame]:
        """Walk the slot queue and return all skipped frames that are now unblocked.

        Removes complete spoken slots from the head of the queue, then emits (and
        removes) skipped slots whose preceding spoken slots are all done. Stops at
        the first incomplete spoken slot.

        Args:
            last_word_pts: When provided, skipped frames receive this PTS so they
                appear immediately after the last spoken word in the timeline.

        Returns:
            AggregatedTextFrame(s) ready to be pushed downstream.
        """
        frames: list[Frame] = []
        while self._slots:
            slot = self._slots[0]
            if slot.spoken and slot.complete:
                self._slots.pop(0)
            elif not slot.spoken and not slot.complete:
                slot.frame.append_to_context = True
                slot.frame.transport_destination = slot.transport_destination
                if last_word_pts:
                    slot.frame.pts = last_word_pts
                logger.debug(f"{self._name}: Flushing Aggregated Frame {slot.frame}")
                frames.append(slot.frame)
                slot.complete = True
                self._slots.pop(0)
            else:
                break  # spoken but not yet complete — wait
        return frames

    def peek_remaining_text(self, context_id: str | None = None) -> str:
        """Return the unspoken TTS text of incomplete spoken slots, without mutating state.

        Mirrors the slot selection of :meth:`force_complete` (spoken, not complete,
        matching ``context_id``) but only reads the trackers. Used to detect a
        mid-stream synthesis stall: if an audio context times out while this is
        non-empty, audio for the remaining text never arrived.

        Args:
            context_id: When provided, only slots registered under this context ID
                are inspected. When None, all incomplete spoken slots are.

        Returns:
            Concatenated remaining TTS text across matching slots ("" if none).
        """
        parts: list[str] = []
        for slot in self._slots:
            if context_id is not None and slot.context_id != context_id:
                continue
            if slot.spoken and not slot.complete and slot.tracker:
                remaining = slot.tracker.get_remaining_tts_text()
                if remaining:
                    parts.append(remaining)
        return "".join(parts)

    def force_complete(self, last_word_pts: int, context_id: str | None = None) -> list[Frame]:
        """Force-complete incomplete spoken slots and flush skipped frames.

        Called at the end of an audio context to handle TTS providers that silently drop
        word-timestamp events. Emits a TTSTextFrame for any remaining unspoken text in
        each incomplete slot, marks it complete, then flushes all now-unblocked skipped
        frames.

        Args:
            last_word_pts: PTS of the last received word frame, used as the PTS for
                force-completed frames and forwarded to :meth:`flush`.
            context_id: When provided, only slots registered under this context ID are
                force-completed; slots of other (typically later) contexts keep waiting
                for their own word timestamps. Force-completing them here would emit
                their full text early, and the real word-timestamp events arriving
                later would duplicate it. When None, all incomplete spoken slots are
                force-completed (legacy behavior).

        Returns:
            Combined list of TTSTextFrames (for incomplete spoken slots) and
            AggregatedTextFrames (skipped slots now unblocked), in emission order.
        """
        frames: list[Frame] = []
        for slot in self._slots:
            if context_id is not None and slot.context_id != context_id:
                continue
            if slot.spoken and not slot.complete:
                if slot.tracker:
                    remaining_text = slot.tracker.get_remaining_tts_text()
                    raw_remaining = slot.tracker.get_remaining_llm_text()
                    if raw_remaining and remaining_text and remaining_text not in raw_remaining:
                        logger.warning(
                            f"{self._name} force-complete: raw_remaining {repr(raw_remaining)} "
                            f"does not contain remaining_text {repr(remaining_text)}, discarding"
                        )
                        raw_remaining = None
                    if remaining_text:
                        logger.debug(
                            f"{self._name} force-completing slot with remaining text "
                            f"{repr(remaining_text)}"
                        )
                        frames.append(
                            self._build_word_frame(
                                remaining_text,
                                last_word_pts,
                                slot.context_id,
                                raw_text=raw_remaining,
                                includes_inter_frame_spaces=slot.includes_inter_frame_spaces,
                            )
                        )
                slot.complete = True
        frames.extend(self.flush(last_word_pts=last_word_pts))
        return frames

    def clear(self) -> None:
        """Clear all slots and context metadata (called on interruption/reset)."""
        self._slots.clear()
        self._context_append_to_context.clear()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_active_slot(self, context_id: str | None = None) -> _AggregatedFrameSlot | None:
        """Return the incomplete spoken slot a word should be attributed to.

        When ``context_id`` is given and a live slot carries that exact id, that
        slot wins over simple queue order. Word-timestamp events name the context
        they belong to, so honouring that id is the only correct attribution when
        two contexts are in flight at once.

        That happens routinely once a TTS cache sits in front of the provider: a
        cache-hit utterance is produced locally and completes almost instantly,
        while a preceding cache-miss utterance is still streaming from the
        provider. Both slots are then incomplete simultaneously, and pure queue
        order hands the cache-hit's words to the still-open miss slot. The miss
        slot consumes them until its own llm_text runs out, so the two utterances
        fuse into one context message and the tail of the second is dropped —
        observed in production as
        "お客様情報を確認いたしますので、少々お待ちください。お待たせ", where
        「いたしました。」 was lost even though the audio played in full.

        Falls back to first-incomplete-slot order when no id is supplied or no
        live slot matches, preserving behaviour for single-context turns and for
        services whose word events carry no context id.
        """
        if context_id is not None:
            matching = next(
                (
                    s
                    for s in self._slots
                    if s.context_id == context_id
                    and s.spoken
                    and not s.complete
                    and s.tracker is not None
                ),
                None,
            )
            if matching is not None:
                return matching
        return next(
            (s for s in self._slots if s.spoken and not s.complete and s.tracker is not None),
            None,
        )

    def _retire_slots_before(self, target: _AggregatedFrameSlot) -> list[Frame]:
        """Complete any incomplete spoken slots queued ahead of *target*.

        Used when a word is routed to a matching context that is not at the head
        of the queue. Older slots are finished as far as audio is concerned, so
        marking them complete releases skipped frames waiting behind them and
        keeps :meth:`flush` from stalling on a slot that will never receive
        another word.
        """
        retired = False
        for slot in self._slots:
            if slot is target:
                break
            if slot.spoken and not slot.complete:
                slot.complete = True
                retired = True
        return self.flush() if retired else []

    def _get_next_active_slot(self, current: _AggregatedFrameSlot) -> _AggregatedFrameSlot | None:
        """Return the first incomplete spoken slot with a tracker after *current*."""
        found = False
        for s in self._slots:
            if s is current:
                found = True
                continue
            if found and s.spoken and not s.complete and s.tracker is not None:
                return s
        return None

    def _build_progress_frame(
        self, slot: _AggregatedFrameSlot, pts: int
    ) -> AggregatedTextProgressFrame:
        """Build an AggregatedTextProgressFrame reflecting the current spoken/remaining state of a slot."""
        assert slot.tracker is not None
        frame = AggregatedTextProgressFrame(
            segment_id=slot.frame.id,
            context_id=slot.context_id,
            text=slot.frame.text,
            aggregated_by=slot.frame.aggregated_by,
            accumulated_text=slot.tracker.get_accumulated_user_facing_text(),
            remaining_text=slot.tracker.get_remaining_user_facing_text(strip=False),
        )
        frame.pts = pts
        return frame

    def _build_word_frame(
        self,
        text: str,
        pts: int,
        context_id: str | None,
        raw_text: str | None = None,
        includes_inter_frame_spaces: bool = False,
    ) -> TTSTextFrame:
        """Build a TTSTextFrame with all standard word-timestamp attributes set."""
        frame = TTSTextFrame(text, aggregated_by=AggregationType.WORD)
        frame.pts = pts
        frame.context_id = context_id
        frame.append_to_context = (
            self._context_append_to_context.get(context_id, True)
            if context_id is not None
            else True
        )
        frame.raw_text = raw_text
        frame.includes_inter_frame_spaces = includes_inter_frame_spaces
        return frame
