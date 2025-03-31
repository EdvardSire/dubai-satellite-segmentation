import segmentation_models_pytorch as smp
from pathlib import Path
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

def get_model():
    return smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=6, 
            activation=ACTIVATION,
            )

def load_model(model_path: Path):
    model = get_model()
    model_state = torch.load(model_path,
                             map_location=torch.device('cpu'),
                             weights_only=True)
    model.load_state_dict(model_state) # pyright: ignore
    model.to(DEVICE) # pyright: ignore
    return model


