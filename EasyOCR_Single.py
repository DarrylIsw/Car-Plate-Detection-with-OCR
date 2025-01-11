import os
import cv2
import torch
import easyocr
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/car_plate_detection/exp/weights/best.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area != 0 else 0

def detect_and_plot(image_path):
    # Read and preprocess the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return []

    # Copy for pre-OCR visualization
    pre_ocr_image = image.copy()

    # Create a mask (white) for dimming later
    mask = np.zeros_like(image, dtype=np.uint8)
    mask.fill(255)

    # Run YOLOv5 detection
    results = model(image_path)
    detections = results.xyxy[0].cpu().numpy()  # Predicted bounding boxes

    predicted_texts = []

    for det_idx, det in enumerate(detections):
        x_min, y_min, x_max, y_max, conf, class_id = map(int, det[:6])

        # Draw bounding boxes in red on pre_ocr_image
        cv2.rectangle(pre_ocr_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box

        # Black out bounding box area on the mask for pre-OCR
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

    # Apply the mask to pre-OCR image to dim areas outside bounding boxes
    pre_ocr_image_with_mask = cv2.addWeighted(pre_ocr_image, 0.7, mask, 0.3, 0)

    for det_idx, det in enumerate(detections):
        x_min, y_min, x_max, y_max, conf, class_id = map(int, det[:6])

        # Crop the detected license plate
        cropped_plate = image[y_min:y_max, x_min:x_max]
        if cropped_plate.size > 0:
            # Show the cropped plate for 1 second (optional, can remove if not needed)
            plate_window_title = f"Detected Plate {det_idx + 1}"
            cv2.imshow(plate_window_title, cropped_plate)
            cv2.waitKey(1000)
            cv2.destroyWindow(plate_window_title)

            # OCR recognition
            ocr_result = reader.readtext(cropped_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=0)

            # If OCR finds text, overlay it on the main image
            if ocr_result:
                text = " ".join(ocr_result)
                predicted_texts.append(text)
                cv2.putText(
                    image,
                    text,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

        # Draw bounding box in red on the final image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Blend the mask so everything outside bounding boxes is dimmed for final OCR
    final_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    cv2.imshow('Detected Plates (No OCR, Masked)', pre_ocr_image_with_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2) Show the final image with OCR text and dimmed background
    cv2.imshow('Detected Plates with OCR Results', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return predicted_texts

# Specify the single image path
image_path = 'testocr/image copy 2.png'  # Replace with your image path

# Process the image and plot results
predicted_texts = detect_and_plot(image_path)

# Print OCR results
print("OCR Results:", predicted_texts)
