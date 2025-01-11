# Import necessary libraries
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import cv2
import numpy as np
import pandas as pd

from utils.plots import plot_results  # YOLOv5 plotting utility
from utils.general import check_requirements
from utils.torch_utils import select_device
from detect import run as detect_run
from train import run as train_run
from val import run as val_run
from detect import run as detect_run
import csv

def setup_training(
    data_yaml, project_name, epochs=50, batch_size=16, img_size=640, weights="yolov5s.pt", optimizer="SGD"
):

    # Check for YOLOv5 dependencies
    check_requirements()

    # Set device (CUDA or CPU)
    device = select_device('')  # Auto-select GPU or CPU

    # Set optimizer parameters
    opt = {
        "SGD": {"momentum": 0.937, "weight_decay": 5e-4},
        # "Adam": {"betas": (0.9, 0.999), "weight_decay": 5e-4},
        # "AdamW": {"betas": (0.9, 0.999), "weight_decay": 1e-2},
    }

    # Update the optimizer settings in training
    train_run(
        data=data_yaml,
        imgsz=img_size,
        batch_size=batch_size,
        epochs=epochs,
        weights=weights,
        project=f"runs/train/{project_name}",
        name="exp",
        exist_ok=True,
        save_period=-1,  
        device=str(device),  # Convert device to a string
        optimizer=optimizer,  # Pass optimizer type to train.py
        momentum=opt.get(optimizer, {}).get("momentum", None),
        weight_decay=opt.get(optimizer, {}).get("weight_decay", None),
        betas=opt.get(optimizer, {}).get("betas", None),
    )

def plot_loss_and_accuracy(results_file, save_dir="runs/train/car_plate_detection/exp/plots"):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return

    # Read the results.csv file into a DataFrame
    results = pd.read_csv(results_file)

    # Clean column names by stripping leading and trailing spaces
    results.columns = results.columns.str.strip()

    # Print available columns for debugging
    print("Available columns in results.csv:", results.columns)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define the required columns for losses and metrics
    train_loss_columns = ['train/box_loss', 'train/obj_loss', 'train/cls_loss']
    val_loss_columns = ['val/box_loss', 'val/obj_loss', 'val/cls_loss']
    metric_columns = ['metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']

    # Plot Loss Curves if loss columns are present
    valid_train_loss_columns = [col for col in train_loss_columns if col in results.columns]
    valid_val_loss_columns = [col for col in val_loss_columns if col in results.columns]
    
    if valid_train_loss_columns or valid_val_loss_columns:
        plt.figure(figsize=(12, 6))
        
        # Plot training losses
        for col in valid_train_loss_columns:
            if results[col].sum() > 0:  # Ensure the column has non-zero values
                plt.plot(results['epoch'], results[col], label=f"Train {col.split('/')[-1].replace('_', ' ').title()}")
        
        # Plot validation losses
        for col in valid_val_loss_columns:
            if results[col].sum() > 0:  # Ensure the column has non-zero values
                plt.plot(results['epoch'], results[col], label=f"Val {col.split('/')[-1].replace('_', ' ').title()}")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(save_dir, "training_validation_losses.png")
        plt.savefig(loss_plot_path)  # Save plot
        plt.show()
        print(f"Loss metrics plot saved to {loss_plot_path}")
    else:
        print("No valid loss columns found in results.csv or all values are zero.")

    # Plot Accuracy Metrics if metric columns are present
    valid_metric_columns = [col for col in metric_columns if col in results.columns]
    if valid_metric_columns:
        plt.figure(figsize=(12, 6))
        for col in valid_metric_columns:
            plt.plot(results['epoch'], results[col], label=col.split('/')[-1].replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Accuracy Metrics Over Epochs')
        plt.legend()
        plt.grid(True)
        accuracy_plot_path = os.path.join(save_dir, "accuracy_metrics.png")
        plt.savefig(accuracy_plot_path)  # Save plot
        plt.show()
        print(f"Accuracy metrics plot saved to {accuracy_plot_path}")
    else:
        print("No valid metric columns found in results.csv.")

def evaluate_model_detailed(weights_path, data_yaml, img_size=640, results_file="runs/train/car_plate_detection/exp/results.csv"):
    check_requirements()
    device = select_device('')

    # Run validation
    results = val_run(data=data_yaml, weights=weights_path, imgsz=img_size, device=device, half=False)

    print("Validation results:", results)

    # Extract metrics
    metrics = results[0] if isinstance(results, tuple) and len(results) > 0 else [0] * 4
    mAP50, mAP50_95, precision, recall = metrics[:4]

    # Calculate accuracy
    accuracy = (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics
    print(f"\nEvaluation Metrics:\n"
          f"mAP@0.5: {mAP50:.3f}\n"
          f"mAP@0.5:0.95: {mAP50_95:.3f}\n"
          f"Precision: {precision:.3f}\n"
          f"Recall: {recall:.3f}\n"
          f"Accuracy: {accuracy:.3f}")

    # Plot confusion matrix
    try:
        confusion_matrix = results[1]  # Adjust index if structure changes
        if isinstance(confusion_matrix, np.ndarray) and confusion_matrix.size > 0:
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
            plt.show()
        else:
            print("Confusion matrix is empty or invalid, skipping plot.")
    except Exception as e:
        print(f"Error while plotting confusion matrix: {e}")

    return {
        "mAP@0.5": mAP50,
        "mAP@0.5:0.95": mAP50_95,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }

def log_metrics(metrics, log_path="runs/train/metrics_log.txt"):

    with open(log_path, "a") as f:
        f.write(f"Evaluation Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.3f}\n")
        f.write("\n")
    print(f"Metrics logged to {log_path}")

def profile_inference(weights_path, source_dir, img_size, conf_thresh=0.4):

    # Set device
    device = select_device('')

    # Ensure img_size is a tuple
    if isinstance(img_size, int):
        img_size = (img_size, img_size)  # Convert to tuple if an integer is provided

    # Start timing
    start_time = time.time()

    # Run detection
    detect_run(
        weights=weights_path,
        source=source_dir,
        imgsz=img_size,  # Pass img_size as a tuple
        conf_thres=conf_thresh,
        device=device
    )

    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")

def detect_plates(weights_path, source_dir, img_size=[640, 640], conf_thresh=0.4, max_display=10):

    # Check for YOLOv5 dependencies
    check_requirements()

    # Set device (CUDA or CPU)
    device = select_device('')  # Auto-select GPU or CPU

    # Define save directory
    save_dir = "runs/detect/exp"  # Adjust directory as needed

    # Run detection
    detect_run(
        weights=weights_path,  # Path to the trained weights file
        source=source_dir,  # Path to the images or video for detection
        imgsz=img_size,  # Image size
        conf_thres=conf_thresh,  # Confidence threshold
        device=device,  # Device for detection
        save_txt=False,  # Avoid saving results to text files
        save_conf=False,  # Avoid saving confidence values
        project="runs/detect",  # Directory to save results
        name="exp",  # Experiment name
        exist_ok=True  # Overwrite if experiment exists
    )

    # Path to saved results
    detected_images_dir = os.path.join(save_dir)

    # Get list of detected images
    detected_images = [
        img_name for img_name in os.listdir(detected_images_dir)
        if img_name.endswith((".jpg", ".png"))  # Include only image files
    ]

    # Display a limited number of images
    for idx, img_name in enumerate(detected_images[:max_display]):  # Limit to `max_display` images
        img_path = os.path.join(detected_images_dir, img_name)
        image = cv2.imread(img_path)
        cv2.imshow(f"Detection Result {idx+1}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Results saved to {detected_images_dir}")
    print(f"Displayed {min(len(detected_images), max_display)} images. Total detected: {len(detected_images)}.")

# Remove results-plot.csv usage and unnecessary metric saving
def evaluate_model_detailed(weights_path, data_yaml, img_size=640):
    check_requirements()
    device = select_device('')

    # Run validation
    results = val_run(data=data_yaml, weights=weights_path, imgsz=img_size, device=device, half=False)

    print("Validation results:", results)

    # Extract metrics
    metrics = results[0] if isinstance(results, tuple) and len(results) > 0 else [0] * 4
    mAP50, mAP50_95, precision, recall = metrics[:4]

    # Calculate accuracy
    accuracy = (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics
    print(f"\nEvaluation Metrics:\n"
          f"mAP@0.5: {mAP50:.3f}\n"
          f"mAP@0.5:0.95: {mAP50_95:.3f}\n"
          f"Precision: {precision:.3f}\n"
          f"Recall: {recall:.3f}\n"
          f"Accuracy: {accuracy:.3f}")

    # Plot confusion matrix
    try:
        confusion_matrix = results[1]  # Adjust index if structure changes
        if isinstance(confusion_matrix, np.ndarray) and confusion_matrix.size > 0:
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
            plt.show()
        else:
            print("Confusion matrix is empty or invalid, skipping plot.")
    except Exception as e:
        print(f"Error while plotting confusion matrix: {e}")

    return {
        "mAP@0.5": mAP50,
        "mAP@0.5:0.95": mAP50_95,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }

# Main function
if __name__ == "__main__":
    # Define paths
    data_yaml_path = "C:/Campus Docs/Image Processing/A.I Stuff/UAS-IMG-PROCESSING-V4/yolov5/data.yaml"
    weights_path = "runs/train/car_plate_detection/exp/weights/best.pt"
    test_images_dir = "yolov5/data/plates-1/test/images"

    # Train YOLOv5
    print("Starting YOLOv5 training...")
    setup_training(
        data_yaml=data_yaml_path,
        project_name="car_plate_detection",
        epochs=4,  # Increase epochs for better training
        batch_size=16,
        img_size=640,
        weights="yolov5s.pt",
        optimizer="SGD"  # Choose optimizer: 'SGD', 'Adam', or 'AdamW'
    )
    print("Training completed!")

    # Evaluate model
    print("Evaluating YOLOv5 model...")
    metrics = evaluate_model_detailed(weights_path=weights_path, data_yaml=data_yaml_path, img_size=640)

    # Log metrics
    print("Logging evaluation metrics...")
    log_metrics(metrics)

    # Plot loss and accuracy
    print("Plotting loss and accuracy metrics...")
    results_file = "runs/train/car_plate_detection/exp/results.csv"
    plot_loss_and_accuracy(results_file, save_dir="runs/train/car_plate_detection/exp/plots")

    # Profile inference performance
    print("Profiling inference performance...")
    profile_inference(weights_path=weights_path, source_dir=test_images_dir, img_size=640, conf_thresh=0.4)

    # Run detection
    print("Running detection on test images...")
    detect_plates(weights_path=weights_path, source_dir=test_images_dir, img_size=[640, 640], conf_thresh=0.4)
    print("Detection completed!")