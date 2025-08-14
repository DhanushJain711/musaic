from google import genai
from google.genai import types
import jinja2
import os
import base64
import dotenv

FILE_PATH = "/Users/dhanush/documents/musaic/good-sounds/sound_files/trumpet_ramon_pitch_stability/neumann/0035.wav"
PROMPT_PATH = "musicbench/prompts/trumpet_prompt.jinja2" 

def setup_gemini_client():
    """Set up and configure Gemini 2.5 Pro client"""
    dotenv.load_dotenv()  # Load environment variables from .env file
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client

def load_system_prompt():
    """Load system prompt from Jinja2 template"""
    with open(PROMPT_PATH, 'r') as f:
        template_content = f.read()
    
    template = jinja2.Template(template_content)
    system_prompt = template.render()
    return system_prompt

def encode_audio_file(file_path):
    """Encode audio file to base64 for sending to Gemini"""
    with open(file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
    return encoded_audio

def main():
    """Main function to run the client"""
    try:
        # Set up client
        client = setup_gemini_client()
        
        # Load system prompt
        system_prompt = load_system_prompt()
        
        # Encode audio file
        encoded_audio = encode_audio_file(FILE_PATH)
        
        # Create the request
        print("Sending request to Gemini...")
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt),
            contents=[
                {
                    "role": "user", 
                    "parts": [
                        {"text": "Please analyze this audio recording:"},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": encoded_audio
                            }
                        }
                    ]
                }
            ]
        )
        
        print("Response received from Gemini:")
        # Print the response
        print(response.text)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
