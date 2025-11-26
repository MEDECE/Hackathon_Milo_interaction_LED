import React from 'react';
import { AudioRecorder } from 'react-audio-voice-recorder';

const AudioInput = ({ onRecordingComplete, disabled }) => {
  const handleComplete = async (blob) => {
    console.log('Audio recording complete:', blob, disabled);
    if (!disabled) {
      onRecordingComplete(blob, 'audio');
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', opacity: disabled ? 0.7 : 1 }}>
      <AudioRecorder
        onRecordingComplete={handleComplete}
        disabled={disabled}
        audioTrackConstraints={{
          noiseSuppression: true,
          echoCancellation: true,
        }}
        downloadOnSavePress={false}
        downloadFileExtension="webm"
      />
    </div>
  );
};

export default AudioInput;