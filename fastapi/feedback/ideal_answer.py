import os
from groq import Groq
from typing import Dict, Optional
from fastapi import HTTPException
import unicodedata
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IdealAnswerGenerator:
    def __init__(self):
        self.client = Groq(
            api_key="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",
        )
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    @staticmethod
    def parse_llm_response(response: str) -> Dict:
        """Parse and validate LLM response with unicode support and JSON extraction."""
        # Normalize unicode characters
        normalized_response = unicodedata.normalize('NFKC', response)
        
        try:
            # First attempt: direct JSON parse
            parsed = json.loads(normalized_response)
            
            # Validate required fields
            required_fields = ['ideal_answer', 'user_strengths', 'areas_for_improvement']
            if not all(field in parsed for field in required_fields):
                raise ValueError("Missing required fields in response")
            
            return parsed
            
        except json.JSONDecodeError:
            # Second attempt: extract JSON if wrapped in text
            start = normalized_response.find("{")
            end = normalized_response.rfind("}")
            
            if start != -1 and end != -1 and end > start:
                possible_json = normalized_response[start:end+1]
                try:
                    parsed = json.loads(possible_json)
                    required_fields = ['ideal_answer', 'user_strengths', 'areas_for_improvement']
                    if all(field in parsed for field in required_fields):
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            # If all parsing fails, raise error with details
            logger.error(f"Failed to parse LLM response: {normalized_response[:500]}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse response as valid JSON"
            )

    async def generate_ideal_answer(self, question: str, user_answer: str) -> Dict:
        """
        Generate an grammatically correct answer and comparison with user's answer.
        
        Args:
            question (str): The original question
            user_answer (str): The user's answer to analyze
            
        Returns:
            Dict: Contains parsed ideal answer analysis
        """
        try:
            prompt = f"""
            Question: {question}
            User's Answer: {user_answer}
            
            Please provide:
            1. A corrected grammatical answer to this question
            2. Analysis of what the user did well
            3. Areas where the user's answer could be improved
            
            Format the response ONLY as a JSON with the following structure. THIS STRUCTURE SHOULD BE MAINTAINED STRICTLY:
            {{
                "ideal_answer": "corrected grammatical answer with correct language as question",
                "user_strengths": "what the user did well with correct language as question",
                "areas_for_improvement": "where the answer could be improved with correct language as question"
            }}
            """

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a grammatical assessment expert. Provide analysis only in JSON format with no additional text.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.1,
                model=self.model,
            )            
    
            response = chat_completion.choices[0].message.content
            logger.info(f"LLM raw response: {response[:200]}")
            
            # Parse JSON response (handles wrapped responses too)
            parsed_data = self.parse_llm_response(response)
            
            logger.info(f"Parsed ideal answer successfully")
            
            return {
                "status": "success",
                "data": parsed_data  # ✅ Return parsed dict, not string
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating ideal answer: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate ideal answer: {str(e)}"
            )

# Create a singleton instance
# ideal_answer_generator = IdealAnswerGenerator()