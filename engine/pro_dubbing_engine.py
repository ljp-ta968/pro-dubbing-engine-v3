import asyncio
import os
import nest_asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from engine.models import DubbingSegment, DubbingSentence
from engine.parser import Parser
from engine.translator import Translator
from engine.tts_handler import TTSHandler
from engine.audio_processor import AudioProcessor

class ProDubbingEngine:
    def __init__(self, api_keys: List[str], output_language: str = "my", voice_gender: str = "Male",
                 tolerance: float = 0.3, max_ai_retries: int = 50, max_rpm: int = 9, bitrate: str = "192k"):
        
        nest_asyncio.apply()
        self.parser = Parser()
        self.translator = Translator(api_keys=api_keys, max_rpm=max_rpm)
        self.audio_processor = AudioProcessor(bitrate=bitrate)
        self.tts_handler = TTSHandler(output_language=output_language, voice_gender=voice_gender,
                                      tolerance=tolerance, max_ai_retries=max_ai_retries,
                                      translator=self.translator, audio_processor=self.audio_processor)
        
        self.output_language = output_language
        self.voice_gender = voice_gender
        self.tolerance = tolerance
        self.max_ai_retries = max_ai_retries
        self.max_rpm = max_rpm
        self.bitrate = bitrate
        self.executor = ThreadPoolExecutor(max_workers=5) # For running async in sync context

    async def _run_async_in_thread(self, coro):
        return await asyncio.get_event_loop().run_in_executor(self.executor, lambda: asyncio.run(coro))

    async def translate_script(self, script_content: str, num_workers: int = 5) -> Dict:
        """Step 1: Translate script and reconstruct initial SRT."""
        # Remove timestamps for translation, assuming they are in [HH:MM:SS] format
        lines_without_timestamps = [l.strip() for l in script_content.split("\n") if l.strip()]
        original_segments_for_reconstruction = self.parser.parse_srt(script_content, self.output_language)

        translated_text = await self.translator.translate_batch_parallel(
            "\n".join(lines_without_timestamps), self.output_language, num_workers
        )
        
        # Reconstruct SRT with original timestamps and translated text
        reconstructed_srt_content = self.parser.reconstruct_srt_with_translation(
            original_segments_for_reconstruction, translated_text
        )
        return {"reconstructed_srt_content": reconstructed_srt_content}

    async def group_sentences(self, srt_content: str) -> List[DubbingSentence]:
        """Step 2: Parse SRT and group segments into sentences."""
        segments = self.parser.parse_srt(srt_content, self.output_language)
        segments_with_sentence_ids = self.parser.group_segments_into_sentences(segments)

        # Create DubbingSentence objects from grouped segments
        grouped_sentences: Dict[int, List[DubbingSegment]] = {}
        for seg in segments_with_sentence_ids:
            if seg.sentence_id not in grouped_sentences:
                grouped_sentences[seg.sentence_id] = []
            grouped_sentences[seg.sentence_id].append(seg)
        
        dubbing_sentences = [DubbingSentence(grouped_sentences[sid], sid) for sid in sorted(grouped_sentences.keys())]
        return dubbing_sentences

    async def generate_tts_and_adjust(self, dubbing_sentences: List[DubbingSentence], num_workers: int = 5, status_callback=None) -> List[DubbingSentence]:
        """Step 3: Generate TTS, perform AI rewriting and speed adjustment."""
        output_dir = "./temp_audio"
        os.makedirs(output_dir, exist_ok=True)

        tasks = [self.tts_handler.generate_tts_for_sentence(sentence, output_dir, status_callback) for sentence in dubbing_sentences]
        await asyncio.gather(*tasks)
        
        return dubbing_sentences

    async def merge_and_finalize(self, dubbing_sentences: List[DubbingSentence]) -> Dict:
        """Step 4: Merge audio files and generate final SRT content."""
        # Update original segments with adjusted text from sentences
        final_segments = []
        for sentence in dubbing_sentences:
            for seg in sentence.segments:
                # Ensure adjusted_text is propagated from sentence to segment
                seg.adjusted_text = sentence.adjusted_text 
                final_segments.append(seg)
        
        # Sort final_segments by segment_id to ensure correct order for SRT generation
        final_segments.sort(key=lambda x: x.segment_id)

        # Merge all audio files
        final_audio_path = os.path.join("./temp_audio", "final_dubbed_audio.mp3")
        self.audio_processor.merge_audio_files(final_segments, final_audio_path)

        # Generate final SRT content
        final_srt_content = self.parser.generate_srt_content(final_segments)

        return {
            "final_audio_path": final_audio_path,
            "final_srt_content": final_srt_content,
            "processed_segments": final_segments
        }

    def run_sync(self, coro):
        """Synchronous wrapper for Streamlit compatibility."""
        return asyncio.run(coro)
