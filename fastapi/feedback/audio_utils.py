import os
import logging
from pydub import AudioSegment
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def convert_audio_to_wav(input_path: str) -> str:
    """Convert audio file to WAV format for processing"""
    try:
        if not input_path:
            logger.warning("No input path provided")
            return None

        # Ensure absolute path
        input_path = os.path.abspath(input_path)
        if not os.path.exists(input_path):
            logger.warning(f"Input file not found: {input_path}")
            return None

        # Create output path in same directory
        output_path = os.path.splitext(input_path)[0] + '.wav'
        
        # Skip if already WAV
        if input_path.lower().endswith('.wav'):
            logger.info(f"File is already WAV format: {input_path}")
            return input_path

        # Run conversion in a thread pool
        def convert():
            try:
                audio = AudioSegment.from_file(input_path)
                audio.export(output_path, format='wav')
                logger.info(f"Successfully converted {input_path} to {output_path}")
                return True
            except Exception as e:
                logger.error(f"Conversion error: {e}")
                return False

        success = await asyncio.to_thread(convert)
        return output_path if success else None

    except Exception as e:
        logger.error(f"Error in convert_audio_to_wav: {e}")
        return None