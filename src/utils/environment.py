from pathlib import Path
import os

def get_environment():
    if os.path.exists("/kaggle/input"):
        return "kaggle"

    if os.path.exists("/content"):
        return "colab"

    return "local"