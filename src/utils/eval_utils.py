import torch
from .metrics import compute_metrics
from sklearn.metrics import confusion_matrix


def eval_one_epoch(
    model, dataloader, loss_fn, device, metrics_list=None, return_confusion_matrix=False
):
    """
    Perform a full evaluation over a dataset for one epoch.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test dataset.
        loss_fn (callable): Loss function used to compute the loss.
        device (torch.device): Device on which to run the computations (CPU or GPU).
        metrics_list (list of str, optional): List of metric names to compute.
        return_confusion_matrix (bool, optional): If True, also compute the confusion matrix (in percentages).

    Returns:
        dict: Dictionary containing:
            - "loss": average loss
            - Other metrics as specified in metrics_list
            - "confusion_matrix" (optional, in percentages)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.long()
            outputs = model(X)
            loss = loss_fn(outputs, y)

            total_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(y)

    avg_loss = total_loss / len(dataloader)
    preds = torch.cat(all_preds).argmax(dim=1)
    labels = torch.cat(all_labels)

    metrics_dict = compute_metrics(labels, preds, metrics_list or ["accuracy"])
    metrics_dict["loss"] = avg_loss

    if return_confusion_matrix:
        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        # Convert to percentages row-wise
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, None] * 100
        metrics_dict["confusion_matrix"] = (
            cm_percent.tolist()
        )  # convert to list for JSON

    return metrics_dict
