from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from routers import users
from config.database import init_db
from feedback.feedback_processor import FeedbackProcessor
from feedback.check_correctness import check_answer_correctness
from feedback.ideal_answer import IdealAnswerGenerator
from setupGeneration import generate_assessment_questions
import logging
import os
from audioProcessor import process_audio_file
from typing import Dict, List,Optional
import types
from pydantic import BaseModel, validator

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class TextAnalysisRequest(BaseModel):
    text: str
    question: Optional[str] = None
    audio_file: Optional[str] = None

    @validator('audio_file')
    def validate_audio_file(cls, v):
        if v and not os.path.exists(v):
            if os.path.exists(os.path.join('temp_audio', os.path.basename(v))):
                return os.path.join('temp_audio', os.path.basename(v))
        return v

app = FastAPI()

API_FRONTEND_URL = os.getenv("API_FRONTEND_URL")

class TempFileManager:
    def __init__(self):
        self.temp_file_path = ""

temp_file_manager = TempFileManager()
ideal_answer_generator = IdealAnswerGenerator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FeedbackProcessor
feedback_processor = FeedbackProcessor()

@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your new project!"}

@app.post("/process-audio") 
async def process_audio(file: UploadFile = File(...), language: str = Form(default="English")):
    try:
        # Create a temporary directory if it doesn't exist

       
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(temp_dir, file.filename)

        logger.info(f"Saving audio file to {temp_file_path}")

        
        temp_file_manager.temp_file_path = temp_file_path

        with open(temp_file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Process the audio file (now includes fluency analysis)
        result = await process_audio_file(temp_file_path, language)
        
        # Clean up the temporary file
        # os.remove(temp_file_path)
        
        # Return the processing result (includes transcription and fluency data)
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/analyze-text")
async def analyze_text(request: TextAnalysisRequest):
    logger.info(f"Analyzing text: {request.text}")
    try:
        # Get the audio file path from the temp_audio directory
        audio_file = None
        if os.path.exists("temp_audio/question_0.mp4"):
            audio_file = "temp_audio/question_0.mp4"
            logger.info(f"Found audio file: {audio_file}")
        else:
            logger.warning("Audio file not found in temp_audio directory")

        feedback = await feedback_processor.analyze_text(
            text=request.text,
            question=request.question,
            tempFileName=audio_file
        )
        
        if 'correctness' in feedback:
            logger.info(f"Correctness score: {feedback['correctness'].get('score', 0)}")
            
        return feedback
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-answer")
async def check_answer(data: Dict = Body(...)):
    try:
        question = data.get("question", "")
        answer = data.get("answer", "")
        
        if not question or not answer:
            raise HTTPException(
                status_code=400,
                detail="Both question and answer are required"
            )
            
        # Check answer correctness and get feedback
        result = check_answer_correctness(question, answer)
        return result
        
    except Exception as e:
        logger.error(f"Error checking answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check answer: {str(e)}"
        )

@app.post("/generate-questions")
async def generate_questions(setup_data: Dict = Body(...)) -> Dict[str, List[str]]:
    try:
        questions = await generate_assessment_questions(setup_data)
        return {"questions": questions}
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate questions: {str(e)}"
        )

@app.post("/get-ideal-answer")
async def get_ideal_answer(data: Dict = Body(...)):
    try:
        question = data.get("question", "")
        user_answer = data.get("answer", "")
        
        if not question or not user_answer:
            raise HTTPException(
                status_code=400,
                detail="Both question and user answer are required"
            )
            
        # Generate ideal answer and analysis
        result = await ideal_answer_generator.generate_ideal_answer(question, user_answer)
        return result
        
    except Exception as e:
        logger.error(f"Error generating ideal answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate ideal answer: {str(e)}"
        )

app.include_router(users.router, prefix="/api/users", tags=["users"])