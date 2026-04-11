import os
from dotenv import load_dotenv

load_dotenv()

SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 20))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
