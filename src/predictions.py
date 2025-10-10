import torch
from PIL import Image


def predict_image(
    image_input,
    model: torch.nn.Module,
    transform: callable,
    class_names: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Predict class probabilities for a single image using a trained model.

    Args:
        image_input (str | PIL.Image.Image): Path to the image or PIL.Image.
        model (torch.nn.Module): Loaded PyTorch model.
        transform (callable): Preprocessing transform to apply to the image.
        class_names (list): List of class names to map predicted index to label.
        device (str): Device to run inference on ('cpu' or 'cuda').

    Returns:
        dict: Dictionary with keys 'pred_index', 'pred_class', 'pred_prob', and 'probabilities'.
    """
    try:
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Transform and move to device
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_index = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_index].item()
            pred_class = class_names[pred_index] if class_names else str(pred_index)

        return {
            "pred_index": pred_index,
            "pred_class": pred_class,
            "pred_prob": pred_prob,
            "probabilities": probs.cpu().numpy(),
        }

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None
