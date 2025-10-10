import torch
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V2_Weights,
)

from src.models import EfficientNetModel, MobileNetV2Model


def load_model(
    state_dict_path,
    model_name: str,
    num_classes: int,
    version: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    """
    Load a trained model.
    """
    try:
        if model_name.startswith("efficientnet"):
            if version is None:
                raise ValueError("EfficientNet requires a version (e.g., 'b0', 'b2').")
            model = EfficientNetModel(version=version, num_classes=num_classes)

        elif model_name.startswith("mobilenet"):
            model = MobileNetV2Model(num_classes=num_classes)

        else:
            raise ValueError(f"Unknown model family: {model_name}")

        # Load state dict
        try:
            state_dict = torch.load(state_dict_path, map_location=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"State dict not found at {state_dict_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading state_dict: {e}")

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # If fails try load in model.model
            model.model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def get_model_transforms(
    model_name: str, version: str = None, augmentation: str = None
):
    """
    Return the default torchvision transforms associated with a model,
    optionally adding augmentation.
    """
    try:
        # Base transforms
        if model_name.startswith("efficientnet"):
            if version is None:
                raise ValueError("EfficientNet requires a version (e.g., 'b0', 'b2').")
            version = version.lower()
            if version == "b0":
                transform = EfficientNet_B0_Weights.DEFAULT.transforms()
            elif version == "b2":
                transform = EfficientNet_B2_Weights.DEFAULT.transforms()
            else:
                raise ValueError(f"Unsupported EfficientNet version: {version}")
        elif model_name.startswith("mobilenet"):
            transform = MobileNet_V2_Weights.DEFAULT.transforms()
        else:
            raise ValueError(f"Unknown model family: {model_name}")

        # Optional augmentation
        if augmentation:
            if augmentation.lower() == "trivialaugmentwide":
                transform = transforms.Compose([TrivialAugmentWide(), transform])
            else:
                raise ValueError(f"Unknown augmentation {augmentation}")

        return transform

    except Exception as e:
        print(f"[ERROR] Failed to create transforms: {e}")
        return None
