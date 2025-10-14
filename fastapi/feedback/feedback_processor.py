import os
import json
import re
import logging
from groq import Groq
from typing import Dict, List, Optional
from dotenv import load_dotenv
from .check_correctness import check_answer_correctness
from .vocab_check import analyze_vocabulary
from .get_pause import get_pause_count
from deepgram import Deepgram
from .audio_utils import convert_audio_to_wav

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackProcessor:
    def __init__(self):
        self.client = Groq(
            api_key="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",
        )
        self.deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY", "a82d3695358bb0fa3cb3b5775271378ef3a77907"))

        self.grammar_prompt = """You are a grammar expert. Analyze the given text for grammatical errors, focusing ONLY on:
        - Incorrect verb tenses (e.g., "I goes" instead of "I go")
        - Subject-verb agreement errors
        - Incorrect use of pronouns
        - Incorrect word usage or word choice
        - Run-on sentences or sentence fragments
        - Incorrect preposition usage
        - Spelling errors
        
        DO NOT flag or count:
        - Don't count the capitalization errors as grammar errors
        - Capitalization issues (e.g., 'i' vs 'I')(remember these are not grammar errors)
        - Missing periods at the end of sentences
        - Stylistic choices
        
        Format your response as a JSON object with two fields:
        {
            "error_count": number,
            "errors": [
                {
                    "word": "incorrect_phrase_or_word",
                    "suggestion": "correct_phrase_or_word",
                    "explanation": "brief explanation of the error"
                }
            ]
        }
        
        Return ONLY the JSON object, no additional text."""

        self.pronunciation_prompt = """You are a pronunciation expert. Analyze the given text for potential pronunciation challenges and mistakes, focusing on:
        - Common pronunciation difficulties for non-native speakers
        - Words with silent letters
        - Complex phonetic combinations
        - Stress patterns in multi-syllable words
        - Commonly mispronounced words
        - Sound pairs that are often confused (e.g., "th" vs "d")
        
        Format your response as a JSON object with two fields:
        {
            "error_count": number,
            "errors": [
                {
                    "word": "challenging_word",
                    "phonetic": "phonetic_representation",
                    "explanation": "brief explanation of the pronunciation challenge"
                }
            ]
        }
        
        Return ONLY the JSON object, no additional text."""

    def analyze_fluency(self, text: str) -> Dict:
        """
        Analyze text for fluency by detecting filler words and hesitations.
        Returns a dictionary containing fluency metrics.
        """
        # List of common filler words and hesitation sounds
        filler_patterns = [
            # Hesitation sounds
            r'\b(hmm+|um+|uh+|aaa+|aa+|mmm+|mm+|ah+|er+|erm+|uhm+|uhmm+|uhhuh|uhuh|eh+|huh+|umm+)\b',
            
            # Common verbal fillers
            r'\b(like|you know|basically|actually|literally|sort of|kind of|i mean|you see|right\?|okay\?|so yeah)\b',
            
            # Repetitive starts
            r'\b(so|well|look|listen|see|okay|like|right|yeah|um so|so basically)\b\s+',
            
            # Uncertainty markers
            r'\b(maybe|probably|somewhat|somehow|kind of like|sort of like|i guess|i think|i suppose)\b',
            
            # Time fillers
            r'\b(at the end of the day|when all is said and done|you know what i mean|what im trying to say)\b',
            
            # Redundant phrases
            r'\b(and stuff|and things|and everything|and all that|or something|or whatever)\b',
            
            # Overused transitions
            r'\b(anyway|anyhow|moving on|going back to|coming back to|speaking of)\b'
        ]

        # Combine patterns into one regex
        combined_pattern = '|'.join(filler_patterns)
        
        # Find all matches
        matches = re.finditer(combined_pattern, text.lower())
        
        # Store all filler words with their positions
        filler_words = []
        total_count = 0
        
        for match in matches:
            filler_words.append({
                "word": match.group(),
                "position": match.start(),
                "context": text[max(0, match.start()-20):min(len(text), match.end()+20)]
            })
            total_count += 1

        # Calculate fluency score (100 - deductions)
        # Deduct points based on the frequency of filler words
        words_in_text = len(text.split())
        filler_ratio = total_count / max(1, words_in_text)
        fluency_score = max(0, min(100, 100 - (filler_ratio * 200)))  # Deduct more points for higher filler word density

        return {
            "fluency_score": round(fluency_score, 1),
            "filler_word_count": total_count,
            "filler_words": filler_words,
            "words_analyzed": words_in_text,
            "filler_ratio": round(filler_ratio * 100, 1),
            "feedback": self._generate_fluency_feedback(total_count, words_in_text, fluency_score)
        }

    def _generate_fluency_feedback(self, filler_count: int, total_words: int, fluency_score: float) -> str:
        """Generate feedback message based on fluency analysis."""
        if fluency_score >= 90:
            return "Excellent fluency! Your speech flows naturally with minimal use of filler words."
        elif fluency_score >= 75:
            return "Good fluency overall. Consider reducing the use of filler words slightly to improve clarity."
        elif fluency_score >= 60:
            return "Moderate fluency. Try to be more conscious of filler words and practice speaking with more confidence."
        else:
            return "Your speech contains frequent filler words which may affect clarity. Focus on reducing hesitations and practice speaking with more confidence."

    async def analyze_grammar(self, text: str) -> Dict:
        """
        Analyze text for grammar mistakes using Groq LLM.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.grammar_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this text: {text}",
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.1,
            )

            # Defensive extraction of the model content (handle different response shapes)
            analysis = None
            try:
                # typical structure
                if hasattr(response, "choices") and response.choices:
                    choice0 = response.choices[0]
                    # support .message.content or .text depending on SDK
                    analysis = getattr(getattr(choice0, "message", None), "content", None) or getattr(choice0, "text", None)
                # fallback to string representation
                if not analysis:
                    analysis = str(response)
            except Exception:
                analysis = str(response)

            if not analysis or not str(analysis).strip():
                logger.warning(f"Empty analysis from LLM. Raw response (truncated): {str(response)[:1000]}")
                return {"error_count": 0, "errors": []}

            return self._parse_grammar_response(analysis)

        except Exception as e:
            logger.exception(f"Error in analyze_grammar: {str(e)}")
            return {
                "error_count": 0,
                "errors": []
            }

    async def analyze_pronunciation(self, text: str, audio_file: str = None) -> Dict:
        """
        Analyze pronunciation using audio file with Deepgram API.
        """
        if not audio_file or not os.path.exists(audio_file):
            return {"error_count": 0, "errors": [], "message": "No audio file provided or file not found"}

        try:
            # Read the audio file
            with open(audio_file, 'rb') as audio:
                source = {'buffer': audio, 'mimetype': 'audio/wav'}
                
                # Configure Deepgram request
                options = {
                    "punctuate": True,
                    "model": "general",
                    "language": "en",
                    "detect_language": True,
                    "utterances": True,
                    "numerals": True,
                    "word_confidence": True,  # Get confidence scores for each word
                    "profanity_filter": False,
                    "alternatives": 1
                }

                response = await self.deepgram.transcription.prerecorded(source, options)

                # Extract pronunciation analysis with timing information
                words = response['results']['channels'][0]['alternatives'][0].get('words', [])
                
                # Enhanced pronunciation analysis
                pronunciation_errors = []
                error_count = 0
                total_confidence = 0
                total_words = len(words)

                for word in words:
                    confidence = word.get('confidence', 0)
                    total_confidence += confidence
                    
                    # More stringent confidence threshold for important words
                    confidence_threshold = 0.85 if len(word['word']) > 3 else 0.75
                    
                    if confidence < confidence_threshold:
                        error_count += 1
                        pronunciation_errors.append({
                            "word": word['word'],
                            "confidence": confidence,
                            "timestamp": {
                                "start": word.get('start', 0),
                                "end": word.get('end', 0)
                            },
                            "suggestion": "Review pronunciation of this word",
                            "severity": "high" if confidence < 0.6 else "medium"
                        })

                avg_confidence = total_confidence / total_words if total_words > 0 else 0

                return {
                    "error_count": error_count,
                    "errors": pronunciation_errors,
                    "overall_confidence": avg_confidence,
                    "pronunciation_score": round((avg_confidence * 100), 1),
                    "words_analyzed": total_words,
                    "feedback": self._generate_pronunciation_feedback(avg_confidence, error_count, total_words)
                }

        except Exception as e:
            print(f"Error in audio pronunciation analysis: {str(e)}")
            return {
                "error_count": 0,
                "errors": [],
                "message": f"Error analyzing pronunciation: {str(e)}"
            }

    async def analyze_pauses(self, text: str, tempFileName: str) -> Dict:
        """Analyze text for pauses using the pause count from the audio file."""
        logger.info(f"Starting pause analysis with file: {tempFileName}")
        try:
            if not tempFileName:
                logger.warning("No audio file provided")
                return {
                    "total_pauses": 0,
                    "pause_details": [],
                    "total_pause_duration": 0,
                    "message": "No audio file provided"
                }

            # Ensure absolute path
            audio_path = os.path.abspath(tempFileName)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found at: {audio_path}")
                return {
                    "total_pauses": 0,
                    "pause_details": [],
                    "total_pause_duration": 0,
                    "message": f"Audio file not found at: {audio_path}"
                }

            # Convert audio to WAV if needed
            wav_file = await convert_audio_to_wav(audio_path)
            if not wav_file:
                logger.error("Failed to convert audio file")
                return {
                    "total_pauses": 0,
                    "pause_details": [],
                    "total_pause_duration": 0,
                    "message": "Audio conversion failed"
                }

            # Get the pause analysis
            pause_analysis = get_pause_count(wav_file)
            logger.info(f"Pause analysis completed: {pause_analysis}")
            return pause_analysis

        except Exception as e:
            logger.error(f"Error in pause analysis: {str(e)}")
            return {
                "total_pauses": 0,
                "pause_details": [],
                "total_pause_duration": 0,
                "error": str(e)
            }

    def _parse_grammar_response(self, response: str) -> Dict:
        """
        Parse the LLM response for grammar analysis.
        Handles:
        - dict responses (already parsed)
        - plain JSON string
        - noisy text containing a JSON object (extract between first '{' and last '}')
        - fallback to a safe default if parsing fails
        """
        try:
            # If already parsed
            if isinstance(response, dict):
                return {
                    "error_count": response.get("error_count", 0),
                    "errors": response.get("errors", [])
                }

            # Ensure string
            resp_text = str(response).strip()
            if not resp_text:
                logger.warning("Received empty grammar response to parse.")
                return {"error_count": 0, "errors": []}

            # First attempt: direct JSON parse
            try:
                data = json.loads(resp_text)
                return {
                    "error_count": data.get("error_count", 0),
                    "errors": data.get("errors", [])
                }
            except json.JSONDecodeError:
                # Try to extract a JSON object if the model wrapped it in text
                start = resp_text.find("{")
                end = resp_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    possible_json = resp_text[start:end+1]
                    try:
                        data = json.loads(possible_json)
                        return {
                            "error_count": data.get("error_count", 0),
                            "errors": data.get("errors", [])
                        }
                    except Exception as ex_inner:
                        logger.warning(f"Failed to parse extracted JSON substring: {ex_inner}. Substring (truncated): {possible_json[:500]}")
                # No valid JSON found
                logger.warning(f"Grammar response is not valid JSON. Response (truncated): {resp_text[:500]}")
                return {"error_count": 0, "errors": []}

        except Exception as e:
            logger.exception(f"Error parsing grammar response: {e}")
            return {
                "error_count": 0,
                "errors": []
            }

    def _parse_pronunciation_response(self, response: str) -> Dict:
        """
        Parse the LLM response for pronunciation analysis.
        """
        try:
            # Parse the JSON response
            data = json.loads(response)
            return {
                "error_count": data.get("error_count", 0),
                "errors": data.get("errors", [])
            }
        except Exception as e:
            print(f"Error parsing pronunciation response: {str(e)}")
            return {
                "error_count": 0,
                "errors": []
            }

    async def analyze_text(self, text: str, question: Optional[str] = None, tempFileName: str = '') -> Dict:
        """
        Analyze text for grammar, pronunciation, vocabulary, fluency and answer correctness.
        """
        try:
            grammar_analysis = await self.analyze_grammar(text)
            pronunciation_analysis = await self.analyze_pronunciation(text, tempFileName)
            vocabulary_analysis = analyze_vocabulary(text)
            fluency_analysis = self.analyze_fluency(text)
            pause_analysis = await self.analyze_pauses(text, tempFileName)
            
            # Get correctness analysis if question is provided
            correctness_analysis = None
            if question:
                correctness_analysis = check_answer_correctness(question, text)
                logger.info(f"Correctness analysis completed with scores - Relevance: {correctness_analysis.get('relevance_score', 0)}, Quality: {correctness_analysis.get('quality_score', 0)}")

            feedback = {
                "grammar": grammar_analysis,
                "pronunciation": pronunciation_analysis,
                "vocabulary": vocabulary_analysis,
                "fluency": fluency_analysis,
                "pauses": pause_analysis,
                "correctness": correctness_analysis,
                "text": text
            }

            return feedback

        except Exception as e:
            logger.error(f"Error in analyze_text: {e}")
            return {"error": str(e)}

    def _generate_pronunciation_feedback(self, confidence: float, error_count: int, total_words: int) -> str:
        """Generate feedback message based on pronunciation analysis."""
        if confidence >= 0.9:
            return "Excellent pronunciation! Your speech is very clear and well-articulated."
        elif confidence >= 0.75:
            return "Good pronunciation overall. Keep practicing to improve clarity."
        elif confidence >= 0.6:
            return "Fair pronunciation. Focus on clear articulation and practice challenging words."
        else:
            return "Your pronunciation needs improvement. Consider practicing difficult words and sounds."
