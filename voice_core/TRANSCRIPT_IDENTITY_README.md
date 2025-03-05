## 1. Background: Why Were User Transcriptions Attributed to the Agent?

1. **LiveKit Identifies Speech by Track Owner**  
   When an audio track is published, LiveKit labels the resulting transcriptions with the participant identity of the publisher. If the agent’s code was publishing the user’s microphone track (or always used the agent’s identity for transcripts), the UI naturally labeled all speech as coming from the agent.

2. **STT Code Using the Agent’s Identity**  
   The `VoiceStateManager` and `EnhancedSTTService` were always defaulting to the **agent** participant identity when publishing transcriptions via:
   ```python
   participant_identity = self._room.local_participant.identity
   ```
   or simply using `"user"` as a hard-coded label. As a result, the LiveKit Playground saw all transcriptions as belonging to the agent.

---

## 2. High-Level Fix

### **A. Track the Actual User Identity**

- When the STT service subscribes to a user’s audio track, it now attempts to find the **real** participant identity by:
  1. Checking `track.participant.identity` directly, or  
  2. Falling back to searching `room.remote_participants` by track SID.  
- The STT code stores this identity in a new `_participant_identity` field.

### **B. Publish Transcriptions With That Identity**

- When the STT service finishes recognizing speech, it calls `_publish_transcript_to_ui(...)`.
- That method calls `VoiceStateManager.publish_transcription(...)`, now passing the **user’s** identity instead of the agent’s identity:
  ```python
  await self.state_manager.publish_transcription(
      text,
      "user",
      is_final=True,
      participant_identity=self._participant_identity
  )
  ```
- `VoiceStateManager.publish_transcription(...)` then uses this identity both in the data channel message (the JSON that says `"type": "transcript", "participant_identity": ...`) and in the Transcription API object (`rtc.Transcription(participant_identity=...)`).

### **C. Result**

- The LiveKit Playground sees a transcription message (or Transcription API call) that says **participant_identity="playground-user"** (or whatever the user identity is). Hence, user speech is labeled as user speech.

---

## 3. What Changed in the Code

**Key modifications** in two files:

1. **`voice_core/stt/enhanced_stt_service.py`**  
   - In `process_audio(...)`, we detect the real participant identity from the remote track or from the fallback.  
   - We store it in `self._participant_identity`.  
   - In `_publish_transcript_to_ui(...)`, we pass `self._participant_identity` to the state manager’s `publish_transcription`.

2. **`voice_core/state/voice_state_manager.py`**  
   - In `publish_transcription(...)`, we add a parameter `participant_identity` and ensure it overrides the default local participant identity.  
   - We then pass that identity to the data channel JSON (`"participant_identity": identity_to_use`) and to the Transcription API (`rtc.Transcription(participant_identity=identity_to_use, ...)`).

The net effect is that **the user’s identity** is always used for user transcriptions, ensuring the UI sees it as user speech.

---

## 4. Summary of the “Breakthrough”

- **We needed to figure out** that the root cause was the agent code was not using the user’s identity for transcriptions.  
- **The fix**: capture the user’s actual identity from the remote audio track, then explicitly pass that identity into both the data channel message and the Transcription API.  
- As soon as we do that, LiveKit labels the user’s speech with the correct identity, and the Playground UI shows “User” (or “playground-user”) instead of the agent.

---

**That’s the essence of the breakthrough:** **publish** user speech with **user** identity rather than always using the agent identity. Once the code tracks and uses `_participant_identity` for each transcript, LiveKit properly attributes speech in the Playground.