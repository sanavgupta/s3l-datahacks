import os
from google import genai

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

try:
    response = client.models.generate_content(
        model='gemini-1.5-flash', 
        contents="Say 'Systems Online'"
    )
    print(f"Response from Gemini: {response.text}")
    print("✅ Connection Verified: Your API key is active and authorized.")
except Exception as e:
    print(f"❌ Connection Failed\nError Details: {e}")