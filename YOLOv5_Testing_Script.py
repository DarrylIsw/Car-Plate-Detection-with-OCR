import os
import torch
from utils.general import check_requirements
from utils.metrics import ConfusionMatrix
from utils.torch_utils import select_device
from val import run as val_run

def evaluate_test_metrics(weights_path, data_yaml, img_size=640, conf_thresh=0.4):
    """
    Evaluate the trained model on the test dataset and compute metrics including Accuracy and F1 Score.
    """
    # Check for YOLOv5 dependencies
    check_requirements()

    # Select device (CUDA or CPU)
    device = select_device('')  # Auto-select GPU or CPU

    print("Starting evaluation on the test dataset...")

    # Perform validation on the test dataset
    results = val_run(
        data=data_yaml,  # Path to the dataset YAML file with `test` key defined
        weights=weights_path,  # Path to the trained weights file
        imgsz=img_size,  # Image size
        device=device,  # Device (e.g., 'cuda' or 'cpu')
        conf_thres=conf_thresh,  # Confidence threshold for detections
        half=False  # Use FP16 precision if supported
    )

    # Extract metrics
    if isinstance(results, tuple) and len(results) > 0:
        metrics = results[0]  # Metrics results
        confusion_matrix = results[1]  # Confusion matrix
    else:
        print("No results found. Check test dataset or model configuration.")
        return

    mAP50 = metrics[0]  # mAP@0.5
    mAP50_95 = metrics[1]  # mAP@0.5:0.95
    precision = metrics[2]  # Precision
    recall = metrics[3]  # Recall

    # Calculate additional metrics: Accuracy and F1 Score
    accuracy = (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print evaluation results
    print("\nTesting Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"mAP@0.5: {mAP50:.3f}")
    print(f"mAP@0.5:0.95: {mAP50_95:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1_score:.3f}")

    # Plot confusion matrix if available
    if isinstance(confusion_matrix, ConfusionMatrix):
        print("Plotting confusion matrix...")
        confusion_matrix.plot()
    else:
        print("Confusion matrix not available.")

    return {
        "Precision": precision,
        "Recall": recall,
        "mAP@0.5": mAP50,
        "mAP@0.5:0.95": mAP50_95,
        "Accuracy": accuracy,
        "F1 Score": f1_score
    }

# Example usage:
if __name__ == "__main__":
    # Define paths
    weights_path = "runs\\train\car_plate_detection\exp\weights\\best.pt"
    data_yaml_path = "yolov5/data.yaml"

    # Evaluate the model on the test dataset
    test_metrics = evaluate_test_metrics(
        weights_path=weights_path,
        data_yaml=data_yaml_path,
        img_size=640,
        conf_thresh=0.4
    )
