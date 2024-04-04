import os
import numpy as np
from PIL import Image

def calculate_iou_accuracy(gt_image, pred_image):
    # Invert colors to make labeled area (foreground) black and background white
    #gt_image = 255 - gt_image
    #pred_image = 255 - pred_image

    # Initialize metrics
    intersection = np.sum(np.logical_and(gt_image, pred_image))
    union = np.sum(np.logical_or(gt_image, pred_image))
    correct_pixels = np.sum(gt_image == pred_image)
    total_pixels = gt_image.size

    # Check if both images are empty
    if np.sum(gt_image) == 0 and np.sum(pred_image) == 0:
        iou = np.nan  # Set IoU to NaN to indicate undefined
    else:
        # Calculate IoU only if union is greater than 0
        if union > 0:
            iou = intersection / union
        else:
            iou = np.nan  # Set IoU to NaN to indicate undefined

    pixel_accuracy = correct_pixels / total_pixels

    return iou, pixel_accuracy

def calculate_precision_recall(gt_image, pred_image):
    # True Positives (TP): Predicted foreground pixels that are also foreground in ground truth
    tp = np.sum(np.logical_and(pred_image == 255, gt_image == 255))

    # False Positives (FP): Predicted foreground pixels that are background in ground truth
    fp = np.sum(np.logical_and(pred_image == 255, gt_image == 0))

    # False Negatives (FN): Background pixels that are foreground in ground truth
    fn = np.sum(np.logical_and(pred_image == 0, gt_image == 255))

    # True Negatives (TN): Background pixels that are also background in ground truth (unused)
    tn = np.sum(np.logical_and(pred_image == 0, gt_image == 0))

    # Calculate Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Calculate Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

def calculate_f1_score(precision, recall):
    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

def main():
    gt_folder = "/home/kai/Desktop/masks"
    pred_folder = "/home/kai/Desktop/output"

    gt_files = os.listdir(gt_folder)
    pred_files = os.listdir(pred_folder)

    # Initialize lists to store metrics
    iou_values = []
    pixel_accuracy_values = []
    precision_values = []
    recall_values = []
    f1_score_values = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_image = np.array(Image.open(os.path.join(gt_folder, gt_file)).convert('L'))  # Convert to grayscale
        pred_image = np.array(Image.open(os.path.join(pred_folder, pred_file)).convert('L'))  # Convert to grayscale

        iou, pixel_accuracy = calculate_iou_accuracy(gt_image, pred_image)
        iou_values.append(iou)
        pixel_accuracy_values.append(pixel_accuracy)

        precision, recall = calculate_precision_recall(gt_image, pred_image)
        precision_values.append(precision)
        recall_values.append(recall)

        f1_score = calculate_f1_score(precision, recall)
        f1_score_values.append(f1_score)

    # Calculate average metrics
    avg_iou = np.nanmean(iou_values)
    avg_pixel_accuracy = np.mean(pixel_accuracy_values)
    avg_precision = np.mean(precision_values)
    avg_recall = np.mean(recall_values)
    avg_f1_score = np.mean(f1_score_values)

    print("Average IoU:", avg_iou)
    print("Average Pixel Accuracy:", avg_pixel_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1_score)

if __name__ == "__main__":
    main()
