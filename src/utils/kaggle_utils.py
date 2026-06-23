import os
import kagglehub

from src.config import KaggleConfig
from src.utils.time_utils import time_now_ist

def upload_to_kagglehub(model,trainer):
    try:
        best_model_path=trainer.checkpoint_callback.best_model_path

        if not best_model_path:
            print("No best model found. Skipping upload to Kagglehub.")
            return

        #handle
        VARIATION=model.__class__.__name__.lower()
        handle =  f"{KaggleConfig.KAGGLE_USERNAME}/{KaggleConfig.KAGGLEHUB_MODEL_REPO}/pytorch/{VARIATION}"

        model_ref = kagglehub.model_upload(
            handle=handle,
            local_model_dir=best_model_path,
            version_notes=f"{model.__class__.__name__} trained on {trainer.current_epoch} epochs and val_map3 {trainer.checkpoint_callback.best_model_score:.4f} at {time_now_ist()}",
        )

        print(f"{VARIATION} Model uploaded to Kagglehub Successfully!")
    except Exception as e:
        print(f"Error uploading model to Kagglehub: {e}")


def kagglehub_download_model(model_handle, ckpt_name):
    download_dir="downloads"

    # Check if already downloaded
    local_path = os.path.join(download_dir, ckpt_name)
    if os.path.exists(local_path):
        print(f"Model already available at: {local_path}")
        return local_path

    try:
        print(f"Downloading model from KaggleHub: {model_handle}")
        model_path = kagglehub.model_download(model_handle,output_dir=download_dir)

        local_path = os.path.join(model_path, ckpt_name)
        print(f"Model downloaded to: {local_path}")
    except Exception as e:
        print(f"Error downloading model from KaggleHub: {e}")
        local_path = None
    return local_path

def load_kagglehub_model(model_handle, ckpt_name, model_class, load_to_device):
    model_path = kagglehub_download_model(model_handle, ckpt_name)
    if model_path:
        model = model_class.load_from_checkpoint(model_path).to(load_to_device)
        model.eval()
        return model
    else:
        print("Failed to load model from KaggleHub.")
        return None