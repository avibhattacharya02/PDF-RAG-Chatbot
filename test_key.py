import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

try:
    # Initialize the client (it looks for GEMINI_API_KEY automatically)
    client = genai.Client()

    # Send a lightweight test request
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Say 'API key is working!' if you can read this.",
    )

    print("Success! Response from model:")
    print(response.text)

except Exception as e:
    print("API test failed:")
    print(str(e))