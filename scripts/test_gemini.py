
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
    raise ValueError(
        "Gemini API key missing. Add GEMINI_API_KEY to your .env file."
    )

print("Connecting to Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3,
)

question = (
    "Suggest three beginner-friendly career pathways "
    "for a woman in India who has completed Class 12."
)

try:
    response = llm.invoke(question)
    print("\nQUESTION:")
    print(question)
    print("\nGEMINI RESPONSE:")
    print(response.content)

except Exception as error:
    print("\nGemini API request failed.")
    print(type(error).__name__)
    print(str(error))
