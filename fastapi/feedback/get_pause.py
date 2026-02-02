import os
import logging
import numpy as np
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pause_count(audio_path: str, threshold_seconds=0.8, amplitude_threshold=0.015):
    """Analyze audio file for pauses."""
    try:
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return {
                "total_pauses": 0,
                "pause_details": [],
                "total_pause_duration": 0,
                "error": "Audio file not found"
            }

        logger.info(f"Loading audio file: {audio_path}")
        # Load audio file
        try:
            audio, sample_rate = librosa.load(audio_path, sr=None)
            logger.info("Successfully loaded audio file")
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return {
                "total_pauses": 0,
                "pause_details": [],
                "total_pause_duration": 0,
                "error": f"Error loading audio: {str(e)}"
            }

        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Calculate the time resolution
        hop_length = 512
        
        # Extract the envelope using RMS energy (more reliable than onset strength)
        audio_envelope = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Convert envelope to time in seconds
        times = librosa.times_like(audio_envelope, sr=sample_rate, hop_length=hop_length)
        
        # Detect periods of silence
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i in range(len(audio_envelope)):
            # Check if current segment is below amplitude threshold
            if audio_envelope[i] < amplitude_threshold:
                if not in_pause:
                    pause_start = times[i]
                    in_pause = True
            else:
                if in_pause:
                    pause_duration = times[i] - pause_start
                    if pause_duration >= threshold_seconds:
                        pauses.append({
                            'start': round(pause_start, 2),
                            'end': round(times[i], 2),
                            'duration': round(pause_duration, 2)
                        })
                    in_pause = False
        
        # Handle pause at the end of audio
        if in_pause:
            pause_duration = times[-1] - pause_start
            if pause_duration >= threshold_seconds:
                pauses.append({
                    'start': round(pause_start, 2),
                    'end': round(times[-1], 2),
                    'duration': round(pause_duration, 2)
                })

        result = {
            'total_pauses': len(pauses),
            'pause_details': pauses,
            'total_pause_duration': round(sum(p['duration'] for p in pauses), 2)
        }
        
        logger.info(f"Detected {len(pauses)} pauses in audio")
        return result

    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return {
            'total_pauses': 0,
            'pause_details': [],
            'total_pause_duration': 0,
            'error': str(e)
        }