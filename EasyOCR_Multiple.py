import os
import cv2
import torch
import easyocr
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/car_plate_detection/exp/weights/best.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def parse_label_file(label_path, image_width, image_height):
    ground_truths = []
    with open(label_path, 'r') as file:
        for line in file:
            label = line.strip().split()
            class_id = int(label[0])
            x_center = float(label[1]) * image_width
            y_center = float(label[2]) * image_height
            width = float(label[3]) * image_width
            height = float(label[4]) * image_height
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            ground_truths.append((x_min, y_min, x_max, y_max, class_id))
    return ground_truths

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of the intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    # Calculate IoU
    return intersection_area / union_area

def detect_and_evaluate(image_path, label_path):
    """
    Detect license plates, recognize plate text, and evaluate bounding box accuracy.
    """
    # Read and preprocess the input image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Parse ground truth labels
    ground_truths = parse_label_file(label_path, image_width, image_height)

    # Run YOLOv5 detection
    results = model(image_path)
    detections = results.xyxy[0].cpu().numpy()  # Predicted bounding boxes

    # Initialize accuracy calculations
    matches = 0
    predicted_texts = []

    for det in detections:
        x_min, y_min, x_max, y_max, conf, class_id = map(int, det[:6])
        best_iou = 0

        for gt in ground_truths:
            gt_box = gt[:4]
            iou = calculate_iou((x_min, y_min, x_max, y_max), gt_box)
            best_iou = max(best_iou, iou)

        if best_iou >= 0.5:  # IoU threshold
            matches += 1

        # OCR recognition for predicted box
        cropped_plate = image[y_min:y_max, x_min:x_max]
        ocr_result = reader.readtext(cropped_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=0)
        predicted_texts.extend(ocr_result)

    # Calculate accuracy
    accuracy = (matches / len(ground_truths)) * 100 if ground_truths else 0
    return accuracy, predicted_texts

def process_folder(folder_path, output_file):
    """
    Process all images and evaluate accuracy for each image, storing results in a file.
    """
    print('Detection and Recognition Processing Starting...')
    results = []
    images_folder = os.path.join(folder_path, 'images')
    labels_folder = os.path.join(folder_path, 'labels')
    total_accuracy = 0
    image_count = 0

    with open(output_file, 'w') as file:
        file.write("Image Name, Accuracy (%), Predicted Texts\n")  # Write header

        for filename in os.listdir(images_folder):
            if filename.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(images_folder, filename)
                label_path = os.path.join(labels_folder, f"{os.path.splitext(filename)[0]}.txt")

                if os.path.exists(label_path):
                    accuracy, predicted_texts = detect_and_evaluate(image_path, label_path)
                    results.append((filename, accuracy, predicted_texts))
                    total_accuracy += accuracy
                    image_count += 1
                    file.write(f"{filename}, {accuracy:.2f}, {', '.join(predicted_texts)}\n")
                else:
                    file.write(f"{filename}, Label file not found, None\n")
        print('Detection and Recognition Processing Complete...')
        print('File Saved Succesfully...')
        # Calculate overall accuracy
        overall_accuracy = total_accuracy / image_count if image_count > 0 else 0
        file.write(f"\nOverall Accuracy: {overall_accuracy:.2f}%\n")

    return results, overall_accuracy

# Specify the folder containing images and labels
folder_path = 'ocrtest/'  # Replace with your folder path
output_file = 'evaluation_results_test.txt'  # Output file name

# Process the folder and save results to the file
evaluation_results, overall_accuracy = process_folder(folder_path, output_file)

# Notify the user
print(f"Results saved to {output_file}")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
