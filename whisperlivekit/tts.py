"""
Text-to-Speech (TTS) module supporting multiple backends:
- Kokoro ONNX (ef_dora voice) for English text → Spanish speech
- Soprano (ekwek/Soprano-1.1-80M) for Spanish text → English speech
"""

import io
import logging
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy audio array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class KokoroTTS:
    """Kokoro ONNX TTS for English text → Spanish speech using ef_dora voice."""

    def __init__(self):
        self.kokoro = None
        self._load_model()

    def _load_model(self):
        try:
            from kokoro_onnx import Kokoro
            self.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            logger.info("Kokoro ONNX TTS model loaded successfully.")
        except ImportError:
            logger.error(
                "kokoro-onnx is not installed. Install it with: pip install kokoro-onnx"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Kokoro ONNX model: {e}")
            raise

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to Spanish speech using ef_dora voice. Returns WAV bytes."""
        if not self.kokoro or not text.strip():
            return None
        try:
            samples, sample_rate = self.kokoro.create(
                text, voice="ef_dora", speed=1.0, lang="es"
            )
            return _audio_to_wav_bytes(samples, sample_rate)
        except Exception as e:
            logger.error(f"Kokoro TTS synthesis failed: {e}")
            return None


class SopranoTTS:
    """Soprano TTS for Spanish text → English speech using ekwek/Soprano-1.1-80M."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._sample_rate = 22050
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            model_name = "ekwek/Soprano-1.1-80M"
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            logger.info("Soprano TTS model loaded successfully.")
        except ImportError:
            logger.error(
                "transformers is not installed. Install it with: pip install transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Soprano model: {e}")
            raise

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to English speech. Returns WAV bytes."""
        if not self.model or not text.strip():
            return None
        try:
            inputs = self.processor(text=text, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=2048)
            audio = self.processor.batch_decode(outputs, return_audio=True)
            if audio is not None and len(audio) > 0:
                samples = audio[0]
                if isinstance(samples, np.ndarray):
                    return _audio_to_wav_bytes(samples, self._sample_rate)
            return None
        except Exception as e:
            logger.error(f"Soprano TTS synthesis failed: {e}")
            return None


class TTSEngine:
    """Unified TTS engine that selects the appropriate backend based on direction.

    - 'es': English text → Spanish speech (Kokoro ONNX, ef_dora)
    - 'en': Spanish text → English speech (Soprano)
    """

    def __init__(self, tts_voice: str = "es"):
        self.tts_voice = tts_voice
        self.kokoro: Optional[KokoroTTS] = None
        self.soprano: Optional[SopranoTTS] = None

        if tts_voice == "es":
            self.kokoro = KokoroTTS()
        elif tts_voice == "en":
            self.soprano = SopranoTTS()
        else:
            logger.warning(f"Unknown TTS voice direction: {tts_voice}. TTS disabled.")

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech. Returns WAV bytes or None."""
        if self.tts_voice == "es" and self.kokoro:
            return self.kokoro.synthesize(text)
        elif self.tts_voice == "en" and self.soprano:
            return self.soprano.synthesize(text)
        return None
