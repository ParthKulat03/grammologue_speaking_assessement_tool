import streamlit as st
import google.generativeai as genai
import cv2
import os
from PIL import Image
import tempfile
import numpy as np

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyB3NdvuL-7B5N8IyfqwCHUHqNHQ48pduFU"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini Pro Vision model
model = genai.GenerativeModel('gemini-2.5-flash')

def extract_frames(video_path, num_frames=5):
    """Extract frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame intervals
    frame_interval = total_frames // num_frames
    
    for i in range(num_frames):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    return frames

# ...rest of your existing code remains the same...

def get_video_review(frames):
    """Generate review using Gemini Pro Vision"""
    prompt = """You are an expert in analyzing human facial expressions and body language.
Analyze the expressions of the person speaking in these video frames.
Provide a detailed analysis based on the following parameters:
1.  **Confidence:** Does the person appear confident and self-assured? Look for signs like a steady gaze, relaxed facial muscles, and genuine smiles.
2.  **Engagement:** How well does the person connect with the viewer through their expressions? Are they making eye contact (towards the camera)? Do their expressions seem authentic and engaging?
3.  **Emotional Expression:** What emotions are being conveyed? Are they appropriate for the context of someone speaking/presenting? Is there a good range of expression, or is it monotonous?
4.  **Clarity of Expression:** Are the expressions clear and easy to interpret, or are they ambiguous?

After your detailed analysis, provide a final judgment and a score out of 100 for their overall expressiveness and presentation skills based on these visual cues. Structure your response with the detailed analysis first, followed by the final score."""
    
    try:
        response = model.generate_content([prompt] + frames)
        return response.text
    except Exception as e:
        return f"Error generating review: {str(e)}"

def main():
    st.title("Facial Expression Analysis")
    st.write("Upload a video of a person speaking to analyze their facial expressions and get a score out of 100.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        try:
            with st.spinner("Analyzing video..."):
                # Extract frames from video
                frames = extract_frames(video_path)
                
                # Get review from Gemini
                review = get_video_review(frames)
                
                # Display results
                st.subheader("Expression Analysis and Score:")
                st.write(review)
                
                # Display extracted frames
                st.subheader("Analyzed Frames:")
                cols = st.columns(len(frames))
                for idx, (col, frame) in enumerate(zip(cols, frames)):
                    col.image(frame, caption=f"Frame {idx+1}", use_column_width=True)
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(video_path)

if __name__ == "__main__":
    main()