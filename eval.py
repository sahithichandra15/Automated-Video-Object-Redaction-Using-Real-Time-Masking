import numpy as np

# --- CONFIGURATION ---
PREDICTIONS_FILE = "my_predictions.txt"
GROUND_TRUTH_FILE = "OTB100/Trans/groundtruth_rect.txt" # You download this with the dataset

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two boxes [x, y, w, h]"""
    # Convert to [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    # Compute both areas
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute IoU
    return interArea / float(boxAArea + boxBArea - interArea)

def calculate_cle(boxA, boxB):
    """Calculates Center Location Error (Euclidean distance)"""
    centerA_x = boxA[0] + boxA[2] / 2.0
    centerA_y = boxA[1] + boxA[3] / 2.0
    
    centerB_x = boxB[0] + boxB[2] / 2.0
    centerB_y = boxB[1] + boxB[3] / 2.0
    
    dist = np.sqrt((centerA_x - centerB_x)**2 + (centerA_y - centerB_y)**2)
    return dist

def evaluate_tracker():
    print("📊 Evaluating Tracker Performance...")
    
    # Load files (handling different delimiters like commas or spaces)
    try:
        preds = np.loadtxt(PREDICTIONS_FILE, delimiter=',')
    except:
        preds = np.loadtxt(PREDICTIONS_FILE) # Try space separated
        
    try:
        gts = np.loadtxt(GROUND_TRUTH_FILE, delimiter=',')
    except:
        gts = np.loadtxt(GROUND_TRUTH_FILE)

    # Ensure lengths match
    min_len = min(len(preds), len(gts))
    preds = preds[:min_len]
    gts = gts[:min_len]

    ious = []
    cles = []
    
    success_threshold = 0.5  # Standard IoU threshold
    precision_threshold = 20 # Standard Pixel Distance threshold
    
    success_count = 0
    precision_count = 0

    for i in range(min_len):
        p_box = preds[i]
        g_box = gts[i]
        
        # If tracker lost the object (0,0,0,0)
        if p_box[2] == 0 or p_box[3] == 0:
            ious.append(0.0)
            # CLE is effectively infinite if lost
            cles.append(999.0) 
            continue
            
        # Calculate metrics
        iou = calculate_iou(p_box, g_box)
        cle = calculate_cle(p_box, g_box)
        
        ious.append(iou)
        cles.append(cle)
        
        if iou > success_threshold: success_count += 1
        if cle < precision_threshold: precision_count += 1

    # Calculate Final Stats
    mean_iou = np.mean(ious)
    mean_cle = np.mean(cles)
    success_rate = (success_count / min_len) * 100
    precision_rate = (precision_count / min_len) * 100

    print("\n--- 🏆 EVALUATION RESULTS ---")
    print(f"Total Frames Analyzed: {min_len}")
    print(f"1. Mean IoU (Overlap): {mean_iou:.3f}")
    print(f"2. Success Rate (IoU > 0.5): {success_rate:.1f}%")
    print(f"3. Mean Center Error: {mean_cle:.1f} pixels")
    print(f"4. Precision Rate (Error < 20px): {precision_rate:.1f}%")
    print("------------------------------")
    
    return ious, cles

if __name__ == "__main__":
    evaluate_tracker()