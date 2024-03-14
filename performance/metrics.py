import numpy as np
from PIL import Image



def calculate_metrics(gt_image, pred_image):
    

    # Invert colors to make labeled area (foreground) black and background white
    gt_image = 255 - gt_image
    pred_image = 255 - pred_image

    # Initialize metrics
    intersection = np.sum(np.logical_and(gt_image, pred_image))
    union = np.sum(np.logical_or(gt_image, pred_image))
    correct_pixels = np.sum(gt_image == pred_image)
    total_pixels = gt_image.size

    # Calculate metrics
    iou = intersection / union if union > 0 else 0
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



# Example usage:
ground_truth_path = "/home/kai/Desktop/evaluate/annotade.png"
predicted_path = "/home/kai/Desktop/evaluate/predicted.png"

# Load ground truth and predicted images
gt_image = np.array(Image.open(ground_truth_path))
pred_image = np.array(Image.open(predicted_path))

iou, pixel_accuracy = calculate_metrics(gt_image, pred_image)
print("IoU:", iou)
print("Pixel Accuracy:", pixel_accuracy)
precision, recall = calculate_precision_recall(gt_image, pred_image)
f1_score = calculate_f1_score(precision, recall)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)