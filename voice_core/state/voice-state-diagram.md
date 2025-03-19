Below is a **single, comprehensive PlantUML sequence diagram** that follows the typical user interaction flow through the `VoiceStateManager` while highlighting:

1. **Initial LISTENING**  
2. **User Transcript → PROCESSING**  
3. **Generating a Response → SPEAKING**  
4. **User Interruption → INTERRUPTED**  
5. **Timeout/Error Detection → ERROR**  
6. **Recovery → Return to LISTENING**  

It also illustrates the **asynchronous** nature of:
- The `_monitor_state` background task for timeouts  
- The `_publish_with_retry` interactions with **LiveKit**  
- The **transcript deduplication** checks  
- The usage of **interrupt events** for TTS interruptions  

> **Tip**: In a single diagram, concurrency can get visually dense. This UML emphasizes the main states and transitions. You can tweak or split it for clarity if needed.

---

```plantuml
@startuml
title VoiceStateManager – Comprehensive Flow

' === Participants / Swimlanes ===
actor User as "User/Client"
participant VoiceStateManager as "VoiceStateManager"
participant StateMonitorTask as "Background\n_monitor_state Task"
participant PublishWithRetry as "_publish_with_retry()"
participant LiveKit as "LiveKit (Room)"
participant TranscriptHandler as "Transcript\nHandler (external)"
participant TTS as "TTS Service (external)"

' We'll show concurrency with "par" or "loop" blocks for the monitor task,
' and normal message flow for user-driven interactions.

autonumber

' =====================================================
'   1. START: VoiceStateManager Already in LISTENING
' =====================================================
User -> VoiceStateManager: (Idle) System is in LISTENING
note over VoiceStateManager
  Current state = LISTENING
  The system is waiting for user transcripts.
end note

' Show the background state monitor running in parallel
par _monitor_state loop:
  StateMonitorTask -> StateMonitorTask: Periodic checks (every 1s)
  alt If current_state in {PROCESSING, SPEAKING, ERROR} too long
    StateMonitorTask -> VoiceStateManager: transition_to(LISTENING)\n(reason=timeout)
    VoiceStateManager -> VoiceStateManager: _cancel_active_tasks()
  else No timeout
  end
  end
else Main Flow:
  ' Continue with the main user interactions
end

' ===============================================
'   2. USER SPEECH DETECTED => PROCESSING
' ===============================================
User -> VoiceStateManager: handle_stt_transcript("Hello Assistant")
activate VoiceStateManager

' 2.1 Check dedup logic
VoiceStateManager -> VoiceStateManager: _is_duplicate_transcript("Hello...")?
alt Transcript is new
  note right
    Not in recent_processed_transcripts,\npassed time threshold => proceed
  end note
else Transcript is duplicate
  note right
    System logs: "Ignoring duplicate transcript"
    If not in LISTENING, forcibly transition to LISTENING
    Return without further processing
  end note
  VoiceStateManager --> User: (Ignored or small back transition to LISTENING)
  deactivate VoiceStateManager
  return
end

' 2.2 Publish user transcript
VoiceStateManager -> VoiceStateManager: store transcript in _recent_processed_transcripts
VoiceStateManager -> PublishWithRetry: publish_transcription("Hello Assistant", sender="user")
note right
  1) Data channel => JSON {type: "transcript", text, etc.}
  2) If track SID found => publish via Transcription API
end note
activate PublishWithRetry
PublishWithRetry -> LiveKit: publish_data(...)
LiveKit --> PublishWithRetry: success/fail
PublishWithRetry --> VoiceStateManager: returns True/False
deactivate PublishWithRetry

' 2.3 Transition to PROCESSING
VoiceStateManager -> VoiceStateManager: transition_to(PROCESSING)
alt Transition allowed
  note right
    _state_lock acquired, old_state=LISTENING -> new_state=PROCESSING
  end note
  VoiceStateManager -> PublishWithRetry: (async) Publish state_update
  PublishWithRetry -> LiveKit: publish_data(...)
  LiveKit --> PublishWithRetry: success
  PublishWithRetry --> VoiceStateManager: done
else Same state or blocked
  note right
    If already PROCESSING, skip or handle concurrency
  end note
end

' 2.4 Invoke external transcript_handler
alt _transcript_handler is set
  VoiceStateManager -> TranscriptHandler: (async) handle transcript "Hello Assistant"
  note over TranscriptHandler
    Possibly calls an LLM, does further logic, etc.
  end note
  group Timeout or success
    TranscriptHandler --> VoiceStateManager: returns successfully
    VoiceStateManager -> VoiceStateManager: finish_processing() => transition_to(LISTENING)
    note right
      Once done, manager auto-transitions back to LISTENING
    end note
  end
else No transcript_handler
  note right
    "No transcript handler registered" -> back to LISTENING
  end note
  VoiceStateManager -> VoiceStateManager: transition_to(LISTENING, reason="no_handler")
end
deactivate VoiceStateManager

' ===================================================
'   3. TTS SPEAKING (Assistant Responds)
' ===================================================
User -> VoiceStateManager: start_speaking(tts_task)
activate VoiceStateManager
VoiceStateManager -> VoiceStateManager: Cancel any existing TTS tasks if needed
VoiceStateManager -> VoiceStateManager: transition_to(SPEAKING)
VoiceStateManager -> PublishWithRetry: publish state_update (SPEAKING)
PublishWithRetry -> LiveKit: publish_data(...)
LiveKit --> PublishWithRetry: success
deactivate PublishWithRetry

' 3.1 TTS begins speaking
VoiceStateManager -> TTS: (async) Speak the final text
note over TTS
  TTS streams audio or uses a TTS engine.\nMight block or run in an async task.
end note

' ===============================
'   4. USER INTERRUPTS
' ===============================
alt User speaks while in SPEAKING
  User -> VoiceStateManager: handle_user_speech_detected("User interruption")
  note over VoiceStateManager
    Detect user speech => set interrupt event,\nCancel current TTS
  end note
  VoiceStateManager -> VoiceStateManager: _interrupt_requested_event.set()
  VoiceStateManager -> TTS: cancel speaking
  TTS --> VoiceStateManager: on_cancelled

  alt If text was provided => PROCESSING
    VoiceStateManager -> VoiceStateManager: transition_to(PROCESSING)
  else => LISTENING
    VoiceStateManager -> VoiceStateManager: transition_to(LISTENING, reason="interrupted")
  end
else No interruption
  note right
    TTS completes normally => state remains SPEAKING until done
  end note
end

' 4.1 TTS completes
TTS --> VoiceStateManager: TTS done
VoiceStateManager -> VoiceStateManager: if still SPEAKING => transition_to(LISTENING)

deactivate VoiceStateManager

' =========================================
'   5. TIMEOUT OR ERROR => ERROR State
' =========================================
' This can be triggered either by:
' a) _monitor_state detecting we are stuck
' b) an internal exception
' c) _publish_with_retry fails beyond max_retries

StateMonitorTask -> VoiceStateManager: Detected stuck in PROCESSING\n(time_in_state > processing_timeout)
activate VoiceStateManager
VoiceStateManager -> VoiceStateManager: _cancel_active_tasks()
VoiceStateManager -> VoiceStateManager: transition_to(ERROR, reason="timeout")
VoiceStateManager -> PublishWithRetry: publish error + state_update
deactivate VoiceStateManager

' ========================================
'   6. ERROR RECOVERY => Return to LISTENING
' ========================================
par Automatic Recovery
  VoiceStateManager -> VoiceStateManager: await 2s then transition_to(LISTENING, reason="error_recovery")
else Manual Recovery
  User -> VoiceStateManager: handle_stt_transcript(...) or start_speaking(...) => triggers transitions
end

' Once the system is recovered:
VoiceStateManager -> VoiceStateManager: State = LISTENING

@enduml
```

---

## Diagram Highlights

1. **Parallel “_monitor_state”**  
   - Displayed with a `par` block at the top. It loops asynchronously in the background and can force a transition to `LISTENING` (and potentially `ERROR`) if timeouts occur.

2. **Transcript Handling & Dedup**  
   - In the **`handle_stt_transcript`** flow, you see a check for duplicates. If duplicates are detected, the transcript is ignored (or the system transitions back to `LISTENING`).

3. **_publish_with_retry**  
   - Shown as a separate participant `PublishWithRetry`, which calls `LiveKit`.  
   - On each state transition, we asynchronously publish updates. If we exceed retry attempts, that can trigger an error transition.

4. **Speaking & Interruption**  
   - When `start_speaking` is invoked, we transition to **SPEAKING** and call the TTS service.  
   - If the user provides speech in mid-TTS, we handle interruption by cancelling TTS, clearing the interrupt event, and possibly moving to **PROCESSING** if new text is provided.

5. **ERROR State & Recovery**  
   - If tasks or publishing fail, or if `_monitor_state` detects a timeout, we transition to **ERROR**.  
   - After some delay (or manual user action), the system transitions back to **LISTENING**.

6. **Async Task Management**  
   - The diagram includes references to `_cancel_active_tasks()` inside transitions to `ERROR` or forced transitions from `_monitor_state`.  
   - The concurrency locks (like `self._state_lock`) are not explicitly depicted but are implied whenever `transition_to(...)` is invoked (the lock ensures only one transition runs at a time).

---

### Tips for Reading / Maintaining the Diagram

- **Blocks**:  
  - `alt/else` blocks show mutually exclusive paths (e.g., user interruption vs. no interruption).  
  - `par` blocks show concurrency (the background state monitor vs. the main flow).
- **Exiting the Diagram**:  
  - After the final state transitions, you remain in `LISTENING`. The system can loop back for new transcripts or TTS requests.
- **Complex Systems**:  
  - For real-world documentation, you may want to create **sub-diagrams** focusing on narrower topics (e.g., timeouts, TTS interruption, or the transcript handler logic) to avoid overwhelming detail in a single chart.

This diagram should provide a solid overview of how the **VoiceStateManager** orchestrates states and tasks in an asynchronous environment, integrating with **LiveKit**, a **Transcript Handler**, and a **TTS** service, while continuously guarded by a background monitor for timeouts and error conditions.