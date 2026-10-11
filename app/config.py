import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

APP_NAME = "Arogyini Career Assistant"
APP_ENV = os.getenv("APP_ENV", "development")