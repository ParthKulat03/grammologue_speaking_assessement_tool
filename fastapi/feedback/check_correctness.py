import os
import json
import logging
from groq import Groq
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(
    api_key="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",
)

def check_answer_correctness(question: str, answer: str) -> Dict:
    """Check the correctness of an answer for a given question."""
    if not question or not answer:
        return {
            "score": 0,
            "relevance_score": 0,
            "quality_score": 0,
            "feedback": "Question or answer is missing",
            "suggestions": "",
            "remark": "Missing input"
        }

    try:
        prompt = """Analyze this answer based on its relevance to the question and quality of explanation.
        Provide ONLY a JSON response in this exact format:
        {
            "relevance_score": <score_0_to_50>,
            "quality_score": <score_0_to_50>,
            "feedback": "<detailed_feedback_about_both_aspects>",
            "suggestions": "<specific_improvement_suggestions>",
            "remark": "<brief_summary>"
        }
        
        Scoring guidelines:
        - Relevance score (0-50): How well the answer addresses the question
        - Quality score (0-50): Clarity, completeness, and coherence of the explanation
        - Total score will be sum of both scores
        
        Return ONLY the JSON object, no additional text or formatting."""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Respond only with valid JSON."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {answer}\n{prompt}"
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
            
        )

        # Extract JSON from the response
        content = response.choices[0].message.content.strip()
        content = content.replace('```json', '').replace('```', '').strip()
        
        try:
            result = json.loads(content)
            # Calculate total score
            total_score = result.get('relevance_score', 0) + result.get('quality_score', 0)
            
            return {
                "score": total_score,
                "relevance_score": result.get('relevance_score', 0),
                "quality_score": result.get('quality_score', 0),
                "feedback": result.get('feedback', ''),
                "suggestions": result.get('suggestions', ''),
                "remark": result.get('remark', '')
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse correctness response: {e}")
            logger.error(f"Raw response: {content}")
            return {
                "score": 0,
                "relevance_score": 0,
                "quality_score": 0,
                "feedback": "Error parsing analysis",
                "suggestions": "System error occurred",
                "remark": "Processing error"
            }

    except Exception as e:
        logger.error(f"Error in correctness check: {e}")
        return {
            "score": 0,
            "relevance_score": 0,
            "quality_score": 0,
            "feedback": f"Error in analysis: {str(e)}",
            "suggestions": "Unable to analyze answer",
            "remark": "Error occurred during analysis"
        }