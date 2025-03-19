Below is a **detailed UML sequence diagram** illustrating the primary interactions within the `EnhancedSTTService` during initialization, audio processing, optional NemoSTT transcription, data publishing, and cleanup. It follows standard UML notations while highlighting key **async/await** and `asyncio.create_task` calls.  

> **Note**: In a single diagram, concurrency and asynchronous tasks can get visually complex. This diagram shows the main flow and indicates where asynchronous tasks (`create_task`) branch off or run in parallel. Depending on your documentation needs, you may prefer splitting certain details into sub-diagrams.

```plantuml
@startuml
title EnhancedSTTService - Sequence Diagram

skinparam ParticipantPadding 10
skinparam BoxPadding 10
skinparam SequenceMessageAlign center
skinparam NotePadding 5

actor Client as "Caller/External Trigger"
participant StateManager
participant EnhancedSTTService
participant AudioPreprocessor
participant VADEngine
participant StreamingSTT
participant NemoSTT as "NemoSTT (Optional)"
participant TranscriptionPublisher
participant LiveKit as "LiveKit (Room + Data Publishing)"
participant LiveKitIdentityManager as "IdentityManager"

' ===========================
' === 1. Initialization  ===
' ===========================
Client -> EnhancedSTTService: instantiate(...)
note right: Constructor sets up components\n(audio_preprocessor, vad_engine,\ntranscriber, publisher, identity_manager, etc.)
Client -> EnhancedSTTService: initialize() [async]
activate EnhancedSTTService

' 1.1 Initialize internal components
EnhancedSTTService -> StreamingSTT: initialize() [async]
activate StreamingSTT
StreamingSTT --> EnhancedSTTService: return
deactivate StreamingSTT

EnhancedSTTService --> Client: STT service initialized
deactivate EnhancedSTTService

' (Optional) Setting the LiveKit Room
Client -> EnhancedSTTService: set_room(room)
activate EnhancedSTTService
EnhancedSTTService -> TranscriptionPublisher: set_room(room)
EnhancedSTTService -> LiveKit: create_task(publish_data(stt_initialized))
note right
  Asynchronously publish "stt_initialized"
  event with model/device info to LiveKit.
end note
EnhancedSTTService -> StateManager: create_task(transition_to(LISTENING)) [if not in SPEAKING/PROCESSING]
deactivate EnhancedSTTService

' ========================================
' === 2. Processing Audio (process_audio) ==
' ========================================
Client -> EnhancedSTTService: process_audio(track) [async]
activate EnhancedSTTService
EnhancedSTTService -> EnhancedSTTService: acquire processing_lock
note right
  Ensures only one process_audio
  call runs at a time.
end note

' 2.1 Get participant identity
EnhancedSTTService -> LiveKitIdentityManager: get_participant_identity(track, room)
LiveKitIdentityManager --> EnhancedSTTService: identity

' 2.2 Publish "listening" state
EnhancedSTTService -> LiveKit: create_task(publish_data(listening_state))
note right
  Asynchronously publishes "listening_state" to LiveKit.\nAlso triggers state transition to LISTENING\nif current state is not SPEAKING/PROCESSING.
end note
EnhancedSTTService -> StateManager: create_task(transition_to(LISTENING)) [if needed]

' 2.3 Create rtc.AudioStream from track
EnhancedSTTService -> LiveKit: AudioStream(track)
note right
  Iterates over incoming audio events
end note

' ========================================
' === 3. Processing Each Audio Frame   ===
' ========================================
loop For each audio event in AudioStream
    EnhancedSTTService -> EnhancedSTTService: Check if state == ERROR?
    alt If ERROR
        note right
          Break loop & cleanup
        end note
        break
    end

    EnhancedSTTService -> AudioPreprocessor: preprocess(audio_data, sample_rate)
    activate AudioPreprocessor
    AudioPreprocessor --> EnhancedSTTService: processed_audio, audio_level_db
    deactivate AudioPreprocessor

    EnhancedSTTService -> VADEngine: process_frame(processed_audio, audio_level_db)
    activate VADEngine
    VADEngine --> EnhancedSTTService: vad_result (is_speaking, speech_segment_complete, valid_speech_segment, etc.)
    deactivate VADEngine

    alt vad_result.is_speaking
        note right
          Append processed_audio to internal buffer\nTrack buffer_duration
        end note
        EnhancedSTTService -> EnhancedSTTService: buffer.append(processed_audio)
    end

    alt vad_result.speech_segment_complete && valid_speech_segment
        note right
          A speech segment ended. Prepare to transcribe.
        end note
        
        ' 3.1 Assemble buffered audio
        EnhancedSTTService -> EnhancedSTTService: full_audio = concat(buffer)
        
        ' 3.2 Transcribe with built-in STT
        EnhancedSTTService -> StreamingSTT: transcribe(full_audio, sample_rate) [async]
        activate StreamingSTT
        StreamingSTT --> EnhancedSTTService: transcription_result
        deactivate StreamingSTT
        
        alt NemoSTT is available
            note right
              NemoSTT can be invoked in parallel (asyncio.create_task).
            end note
            EnhancedSTTService -> NemoSTT: create_task(transcribe(full_audio)) [async]
            note right
              NemoSTT transcribes in background.\nOnce complete, result can refine final transcript.
            end note
        end

        alt transcription_result.success
            alt transcription_result.text is not empty
                EnhancedSTTService -> TranscriptionPublisher: publish_transcript(text, identity, is_final=not NemoSTT)
                activate TranscriptionPublisher
                TranscriptionPublisher --> EnhancedSTTService: (ack)
                deactivate TranscriptionPublisher

                alt on_transcript callback exists
                    note right
                      on_transcript can be sync or async.\nIf async, we await it.
                    end note
                    EnhancedSTTService -> on_transcript: text
                end
            end
        end
        
        EnhancedSTTService -> EnhancedSTTService: buffer.clear()
    end
end

' 3.3 Final cleanup after audio ends or error
EnhancedSTTService -> EnhancedSTTService: release processing_lock
EnhancedSTTService --> Client: return None or final transcript

' ========================================
' === 4. Cleanup Procedures            ===
' ========================================
Client -> EnhancedSTTService: cleanup() [async]
activate EnhancedSTTService
EnhancedSTTService -> EnhancedSTTService: stop_processing() [async]
activate EnhancedSTTService
EnhancedSTTService -> EnhancedSTTService: cancel active_task?
EnhancedSTTService -> EnhancedSTTService: clear_buffer()
deactivate EnhancedSTTService

EnhancedSTTService -> LiveKit: create_task(publish_data(stt_cleanup)) [via state_manager._publish_with_retry]
note right
  Publish "stt_cleanup" event
end note

EnhancedSTTService -> StreamingSTT: cleanup() [async]
EnhancedSTTService --> EnhancedSTTService: buffer = [], buffer_duration = 0.0

EnhancedSTTService --> Client: cleanup complete
deactivate EnhancedSTTService

@enduml
```

### Diagram Explanation

1. **Initialization**  
   - The `EnhancedSTTService` is instantiated with references to its modular components (`AudioPreprocessor`, `VADEngine`, `StreamingSTT`, `TranscriptionPublisher`, `LiveKitIdentityManager`).  
   - On `initialize()`, it triggers `StreamingSTT.initialize()` and publishes an initialization event (`stt_initialized`) to LiveKit asynchronously.

2. **Setting the LiveKit Room (optional step)**  
   - `set_room(room)` associates a LiveKit `Room` instance with the service, which is needed for publishing data events and retrieving participant identities.

3. **Processing Audio**  
   - `process_audio(track)` is the core method for handling incoming `rtc.AudioTrack` data from LiveKit.  
   - A lock (`processing_lock`) ensures only one track is processed at a time.  
   - The service retrieves the participant identity (via `LiveKitIdentityManager`), publishes a "listening" event to LiveKit, and transitions the `StateManager` to a `LISTENING` state if appropriate.  
   - An `AudioStream` is created to yield audio frames from the track in an asynchronous loop.

4. **Processing Each Frame**  
   - For each frame, audio is preprocessed (`AudioPreprocessor.preprocess`) and sent to the `VADEngine` to detect speech activity.  
   - If speech is active, audio data is buffered.  
   - When a speech segment completes (based on VAD), the buffered audio is passed to `StreamingSTT` for transcription.  
   - Optionally, if a `NemoSTT` instance is provided, the same segment is sent there in an asynchronous task (`asyncio.create_task`), allowing both transcriptions to run concurrently.  
   - If transcription is successful, the `TranscriptionPublisher` publishes a preliminary or final transcript to LiveKit, and the optional `on_transcript` callback is invoked.

5. **Cleanup**  
   - Once audio processing ends or is explicitly stopped, the service releases the lock and returns control.  
   - A separate `cleanup()` method stops active tasks, clears buffers, optionally publishes a `stt_cleanup` event, and calls `cleanup()` on the transcriber.

---

## Suggestions for Further Clarity

- **Separate Diagrams for Major Phases**: Because the audio processing loop is quite detailed, consider creating one high-level diagram for overall workflow (init, set_room, process_audio, stop/cleanup) and another focusing specifically on the per-frame loop (VAD, buffering, transcription, publishing).  
- **Include NemoSTT Results Handling**: If you handle final NemoSTT results later in the workflow (e.g., replacing or refining the initial transcript), a separate interaction after NemoSTT completes could clarify how that final text is published or updated.  
- **Highlight Error Conditions**: If your application has more complex error recovery, you may want to include an “alt” or “opt” block showing transitions to an `ERROR` state, logging, and specific retry or cleanup behaviors.  

Feel free to let me know if you need any additional interactions, clarifications, or specific method-invocation details added to the diagram!