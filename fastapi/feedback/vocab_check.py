import logging
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Advanced vocabulary words categorized by type
ADVANCED_VOCABULARY = {
    "verbs": {
        "accomplish", "acquire", "advocate", "analyze", "assess", "collaborate",
        "comprehend", "contemplate", "demonstrate", "derive", "elaborate",
        "emphasize", "enhance", "establish", "evaluate", "facilitate",
        "generate", "implement", "incorporate", "investigate", "maintain",
        "optimize", "perceive", "pursue", "resolve", "synthesize", "utilize",
        "articulate", "cultivate", "differentiate", "exemplify", "formulate",
        "hypothesize", "integrate", "leverage", "mediate", "navigate",
        "orchestrate", "quantify", "rationalize", "streamline", "validate"
    },
    "adjectives": {
        "abundant", "adequate", "comprehensive", "crucial", "diverse",
        "efficient", "fundamental", "innovative", "integral", "optimal",
        "precise", "prominent", "robust", "significant", "sophisticated",
        "substantial", "versatile", "analytical", "cohesive", "empirical",
        "holistic", "imperative", "methodical", "nuanced", "paramount",
        "pragmatic", "quintessential", "systematic", "theoretical"
    },
    "adverbs": {
        "accordingly", "consequently", "effectively", "extensively",
        "fundamentally", "predominantly", "precisely", "primarily",
        "significantly", "substantially", "thoroughly", "analytically",
        "coherently", "decisively", "empirically", "intrinsically",
        "methodically", "objectively", "pragmatically", "strategically"
    },
    "transitions": {
        "additionally", "consequently", "furthermore", "however",
        "moreover", "nevertheless", "subsequently", "therefore",
        "alternatively", "comparatively", "conversely", "correspondingly",
        "essentially", "ultimately", "notwithstanding", "similarly",
        "specifically", "whereas"
    },
    "academic": {
        "analysis", "approach", "concept", "context", "data",
        "evidence", "framework", "hypothesis", "methodology",
        "perspective", "principle", "process", "research", "theory",
        "algorithm", "correlation", "discretion", "paradigm",
        "parameter", "phenomenon", "protocol", "synthesis",
        "threshold", "variable", "velocity", "criterion"
    }
}

def analyze_vocabulary(text: str) -> Dict:
    """
    Analyze the vocabulary usage in the given text.
    """
    try:
        words = set(text.lower().split())
        found_words = {category: [] for category in ADVANCED_VOCABULARY}
        total_advanced_words = 0
        
        for category, word_set in ADVANCED_VOCABULARY.items():
            for word in word_set:
                if word in words:
                    found_words[category].append(word)
                    total_advanced_words += 1
        
        base_score = 50
        max_expected_words = 10
        additional_score = min(50, (total_advanced_words / max_expected_words) * 50)
        vocabulary_score = base_score + additional_score

        return {
            "vocabulary_score": round(vocabulary_score, 1),
            "total_advanced_words": total_advanced_words,
            "advanced_words_by_category": found_words,
            "unique_advanced_words": list(set().union(*found_words.values())),
            "feedback": generate_vocabulary_feedback(total_advanced_words, vocabulary_score)
        }
    except Exception as e:
        logger.error(f"Error in vocabulary analysis: {e}")
        return {
            "vocabulary_score": 50.0,
            "total_advanced_words": 0,
            "advanced_words_by_category": {},
            "unique_advanced_words": [],
            "feedback": "Error analyzing vocabulary"
        }

def generate_vocabulary_feedback(word_count: int, score: float) -> str:
    """
    Generate feedback message based on vocabulary usage.
    
    Args:
        word_count (int): Number of advanced words used
        score (float): Vocabulary score
        
    Returns:
        str: Feedback message
    """
    if score >= 90:
        return "Excellent vocabulary usage! Your language is sophisticated and varied."
    elif score >= 75:
        return "Good use of advanced vocabulary. Keep expanding your word choices."
    elif score >= 60:
        return "Decent vocabulary range. Try incorporating more varied expressions."
    else:
        return "Consider using more sophisticated vocabulary to enhance your expression."