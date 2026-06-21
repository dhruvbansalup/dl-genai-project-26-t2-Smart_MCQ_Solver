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


def download_from_kagglehub(model_handle):
    pass