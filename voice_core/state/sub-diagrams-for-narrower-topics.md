Below are **three smaller PlantUML sequence diagrams** (sub-diagrams) focused on narrower topics in the `VoiceStateManager` and the `InterruptibleTTSService`. Each diagram highlights a distinct aspect:

1. **Timeout Logic** – How the background `_monitor_state` detects stuck states, cancels tasks, and transitions to `ERROR` or back to `LISTENING`.  
2. **TTS Interruption** – How the user interrupts TTS mid-speech, how events are set, how tasks are canceled, and how states transition.  
3. **Transcript Handler Flow** – How a new user transcript is processed, deduplicated, published, handed to the external transcript handler, and possibly leads to generating TTS responses.

Use these sub-diagrams alongside your main comprehensive diagram to clarify specific behaviors in isolation.

---

## 1) **Timeout Logic** 

```plantuml
@startuml
title Sub-Diagram: Timeout Logic in _monitor_state

participant "VoiceStateManager" as VSM
participant "Background Task" as Monitor
participant "User Tasks" as Tasks

autonumber

Monitor -> Monitor: loop every 1 second
note over Monitor
  The _monitor_state task runs in a loop,
  checking how long we've been in a non-LISTENING state.
end note

Monitor -> VSM: current_state = VSM._state
alt State = PROCESSING or SPEAKING or ERROR
  Monitor -> VSM: Check time_in_state
  alt time_in_state > threshold
    Monitor -> VSM: Timeout detected => transition_to(LISTENING/ERROR)
    note over VSM
      1. Cancels any in-progress tasks
      2. If stuck in ERROR, might stay or attempt recovery
    end note
    VSM -> VSM: _cancel_active_tasks()
    group Timed out in PROCESSING
      VSM -> VSM: transition_to(LISTENING, reason="processing_timeout")
    end
    group Timed out in SPEAKING
      VSM -> VSM: transition_to(LISTENING, reason="speaking_timeout")
    end
    group Timed out in ERROR
      VSM -> VSM: Potential forced reset or fallback
    end
  else not timed out
    Monitor -> Monitor: continue loop
  end
else State = LISTENING / IDLE / INTERRUPTED
  Monitor -> Monitor: no timeout checks needed
end

Monitor -> Monitor: (loop repeats)
note right
  Continues until voice manager shuts down
end note

@enduml
```

**Key Points**  
- **`_monitor_state`** runs in an infinite loop (with sleeps in between) to detect if the system is “stuck” in `PROCESSING`, `SPEAKING`, or `ERROR` for too long.  
- On timeout, it triggers `transition_to(LISTENING)` (or stays in `ERROR`), **cancels tasks** (`_cancel_active_tasks()`), and logs a warning.  

---

## 2) **TTS Interruption** 

Below is a more focused look at how the `VoiceStateManager` and the `InterruptibleTTSService` coordinate mid-speech interruption:

```plantuml
@startuml
title Sub-Diagram: TTS Interruption Flow

actor User
participant "VoiceStateManager" as VSM
participant "InterruptibleTTSService" as ITTS
participant "TTS Task" as TTS_Task
participant "VoiceState" as VS

autonumber

== Start TTS ==
User -> VSM: start_speaking(tts_task)
activate VSM
VSM -> VSM: transition_to(SPEAKING)
note over VSM
  _state = SPEAKING
  Publish state updates to LiveKit
end note
VSM -> ITTS: speak(text)
activate ITTS
ITTS -> TTS_Task: (async) _stream_tts(text)
deactivate VSM

== Interruption Detected ==
User -> VSM: handle_user_speech_detected("interrupting text...")
alt Currently in SPEAKING
  note right
    VoiceStateManager sees user transcript => sets `_interrupt_requested_event`
  end note
  VSM -> ITTS: stop()
  activate ITTS
  ITTS -> TTS_Task: cancel()
  note over TTS_Task
    TTS streaming loop is interrupted => CancelledError
  end note
  TTS_Task --> ITTS: TTS_Task done (cancelled)
  deactivate ITTS

  group If user provided new text
    VSM -> VSM: transition_to(PROCESSING, {"text": "interrupting text..."})
    note over VSM
      Cancels prior tasks if needed;
      new transcript is handled
    end note
  else No text => Just return to LISTENING
    VSM -> VSM: transition_to(LISTENING, reason="interrupted")
  end
else Not in SPEAKING
  note right
    If not in SPEAKING, maybe ignore or queue
  end note
end

== TTS Service Cleanup ==
ITTS -> ITTS: _active = False
ITTS -> VSM: transition_to(LISTENING, reason="tts_stopped") [if needed]
ITTS -> VSM: _interrupt_handled_event.set()

@enduml
```

**Key Points**  
- **`start_speaking()`** transitions the manager to `SPEAKING`, then calls `ITTS.speak(...)`.  
- If the **user** speaks while TTS is active, `handle_user_speech_detected(...)` triggers an **interrupt** path:
  - The manager sets `_interrupt_requested_event`, calls `ITTS.stop()`.  
  - The TTS service cancels its internal streaming task, clearing or flushing buffers.  
- Depending on whether the user also provided new text, we go to `PROCESSING` or revert to `LISTENING`.  

---

## 3) **Transcript Handler Flow** 

This sub-diagram illustrates how a **new transcript** is processed, including **deduplication**, publishing to LiveKit, and handing off to an **external transcript handler** (often responsible for LLM calls).

```plantuml
@startuml
title Sub-Diagram: Transcript Handler Flow

actor User
participant "VoiceStateManager" as VSM
participant "TranscriptHandler" as TH
participant "PublishWithRetry" as PWR

autonumber

== New User Transcript Arrives ==
User -> VSM: handle_stt_transcript("Hello Assistant")
activate VSM

' 1. Deduplication
VSM -> VSM: _is_duplicate_transcript("Hello Assistant")?
alt If duplicate
  note right
    VSM logs "Ignoring duplicate"
    Possibly transition_to(LISTENING) if not in LISTENING
  end note
  VSM --> User: return False
  deactivate VSM
  return
else Not duplicate
  VSM -> VSM: _recent_processed_transcripts.append(...)
end

' 2. Publish transcript to UI
VSM -> PWR: publish_transcription("Hello Assistant", sender="user")
activate PWR
PWR -> PWR: attempt 1..max_retries
PWR -> PWR: publish_data to LiveKit
note right
  If success => return True
  If fail => retry or eventually fail => set VSM._publish_stats
end note
PWR --> VSM: success/failure
deactivate PWR

' 3. Check if manager is SPEAKING
alt state == SPEAKING
  note right
    Interpreted as an interruption => TTS cancelled
  end note
  VSM -> VSM: handle_user_speech_detected("Hello Assistant")
  deactivate VSM
  return
else state != SPEAKING
  note right
    Move on to PROCESSING
  end note
end

' 4. Transition to PROCESSING
VSM -> VSM: transition_to(PROCESSING, {"text": "..."} )

' 5. Call external transcript handler
alt _transcript_handler is set
  VSM -> TH: (async) handle("Hello Assistant")
  note over TH
    Possibly calls an LLM or pipeline
    Returns an answer or completes
  end note
  TH --> VSM: done
  VSM -> VSM: finish_processing() => transition_to(LISTENING)
else no transcript handler
  note right
    logs warning => transition_to(LISTENING)
  end note
end
deactivate VSM

@enduml
```

**Key Points**  
- **Deduplication**: `_is_duplicate_transcript()` checks if the text is repeated or within a too-short interval.  
- **Publishing**: `publish_transcription(...)` calls `_publish_with_retry` internally to ensure data and transcription are sent to LiveKit.  
- If currently in **SPEAKING**, the transcript is interpreted as an **interruption** path. Otherwise, we **transition to PROCESSING**.  
- If a **transcript handler** is registered, we create an async task to process it (e.g., run an LLM). On completion or timeout, we revert to **LISTENING**.

---

## How to Use These Sub-Diagrams

- Each sub-diagram is meant to be **referenced alongside** your main, end-to-end flow diagram.  
- They provide a closer look at specific behaviors so you or your team can easily see how **interruptions**, **timeouts**, and **transcript handling** work in isolation.  
- The code references (e.g., `transition_to`, `_cancel_active_tasks`, `_is_duplicate_transcript`, `publish_with_retry`, etc.) align with your Python implementation in `VoiceStateManager` and `InterruptibleTTSService`.  

Feel free to adjust naming, detail level, or messages as needed to best document your specific codebase and usage patterns.