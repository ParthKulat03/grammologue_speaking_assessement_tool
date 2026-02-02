from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_api():
    try:
        # Initialize Groq client with your API key
        client = Groq(
            api_key="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"
        )

        # Test prompt
        test_prompt = "Write a simple hello world message"

        print("Attempting to connect to Groq API...")
        
        # Make API call with correct parameters
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": test_prompt,
                }
            ],
            temperature=0.1,
            max_tokens=100,  # Added to limit response length
        )

        # Print response
        print("\nAPI Response:")
        print("Status: Success")
        print("Model used:", "meta-llama/llama-4-scout-17b-16e-instruct")
        print("Response content:", response.choices[0].message.content)
        return True

    except Exception as e:
        print("\nError occurred:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Groq API test...")
    result = test_groq_api()
    if result:
        print("\nTest completed successfully! You can use the Llama model.")
    else:
        print("\nTest failed! Please check your API key and connection.")