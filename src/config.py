from dotenv import load_dotenv
import os

load_dotenv()

class EnvConfig:
    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
    PROCESSED_DATA_DIR=os.getenv("PROCESSED_DATA_DIR")