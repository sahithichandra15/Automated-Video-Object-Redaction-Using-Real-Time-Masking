import cv2
import numpy as np
import os
import sys
import glob

# --- OTB-100 CONFIGURATION ---
SEQUENCE_NAME = 'Trans'  # Testing on the rotating Coke can!
BASE_DIR = f'OTB100/{SEQUENCE_NAME}'

IMG_FOLDER_PATH = os.path.join(BASE_DIR, 'img')
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, 'groundtruth_rect.txt')
RESULTS_PATH = 'my_predictions.txt'

# --- COLOR GATE SETTINGS ---
# 1.0 is a perfect color match, 0.0 is no match, negative is opposite colors.
COLOR_THRESHOLD = 0.35  

def create_tracker():
    """Robust tracker creation that survives OpenCV version changes."""
    # 1. Try standard CSRT (Most common in opencv-contrib)
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
        
    # 2. Try newer dot-syntax CSRT
    elif hasattr(cv2, 'TrackerCSRT'):
        return cv2.TrackerCSRT.create()
        
    # 3. Fallback 1: KCF Tracker (Faster, usually included)
    elif hasattr(cv2, 'TrackerKCF_create'):
        print("⚠️ CSRT missing. Falling back to KCF Tracker.")
        return cv2.TrackerKCF_create()
        
    # 4. Fallback 2: MIL Tracker (Built into CORE OpenCV, guaranteed to exist)
    elif hasattr(cv2, 'TrackerMIL_create'):
        print("⚠️ CSRT/KCF missing. Falling back to built-in MIL Tracker.")
        return cv2.TrackerMIL_create()
        
    # 5. Ultimate Fallback: MIL dot-syntax
    else:
        print("⚠️ Using built-in dot-syntax MIL Tracker.")
        return cv2.TrackerMIL.create()

def load_dataset_initialization():
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Error: Ground truth not found at {GROUND_TRUTH_PATH}")
        sys.exit(1)
        
    with open(GROUND_TRUTH_PATH, 'r') as f:
        first_line = f.readline().strip()
        
    if ',' in first_line:
        gt_box = [float(v) for v in first_line.split(',')]
    else:
        gt_box = [float(v) for v in first_line.split()]
        
    x = max(0, int(gt_box[0] - 1))
    y = max(0, int(gt_box[1] - 1))
    w = int(gt_box[2])
    h = int(gt_box[3])
    
    return (x, y, w, h)

def get_color_histogram(image, box):
    """Extracts the Color Fingerprint (HSV) of the bounding box"""
    x, y, w, h = box
    roi = image[y:y+h, x:x+w]
    if roi.size == 0: return None
    
    # Convert to HSV (Hue, Saturation, Value) - much better for lighting changes
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate a 2D Histogram based on Hue and Saturation
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def run_color_gated_evaluation():
    print(f"🚀 Launching Color-Gated CSRT Evaluator on: {SEQUENCE_NAME}")
    
    image_files = sorted(glob.glob(os.path.join(IMG_FOLDER_PATH, '*.jpg')))
    if not image_files: return
    total_frames = len(image_files)

    # 1. Read First Frame & Initialize
    first_frame = cv2.imread(image_files[0])
    height, width = first_frame.shape[:2]
    
    init_box = load_dataset_initialization()
    init_x, init_y, init_w, init_h = init_box
    
    # 2. Extract the "Color Fingerprint" of the target from Frame 1
    baseline_hist = get_color_histogram(first_frame, init_box)
    
    tracker = create_tracker()
    tracker.init(first_frame, init_box)
    
    results_file = open(RESULTS_PATH, "w")
    results_file.write(f"{init_x},{init_y},{init_w},{init_h}\n")

    # MAIN LOOP
    for i in range(1, total_frames):
        frame_path = image_files[i]
        frame = cv2.imread(frame_path)
        
        sys.stdout.write(f"\rProcessing Frame {i+1}/{total_frames}")
        sys.stdout.flush()

        display_frame = frame.copy()

        # Let CSRT predict where it thinks the object is
        success, bbox = tracker.update(frame)
        
        target_locked = False

        if success:
            x, y, w, h = [int(v) for v in bbox]
            
            # Clamp bounds
            out_x = max(0, min(x, width - 1))
            out_y = max(0, min(y, height - 1))
            out_w = max(1, min(w, width - out_x))
            out_h = max(1, min(h, height - out_y))
            
            current_box = (out_x, out_y, out_w, out_h)
            
            # --- THE COLOR GUARDRAIL ---
            # What color is inside CSRT's box right now?
            current_hist = get_color_histogram(frame, current_box)
            
            if current_hist is not None:
                # Mathematically compare the current color to the Frame 1 color
                color_similarity = cv2.compareHist(baseline_hist, current_hist, cv2.HISTCMP_CORREL)
                
                if color_similarity >= COLOR_THRESHOLD:
                    target_locked = True
                    
                    # Log the box for the grader
                    results_file.write(f"{out_x},{out_y},{out_w},{out_h}\n")
                    
                    # Draw HUD
                    cv2.rectangle(display_frame, (out_x, out_y), (out_x+out_w, out_y+out_h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"LOCKED (Color: {color_similarity:.2f})", (out_x, out_y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Color failed! It went behind the bush. 
                    # We don't kill CSRT, we just ignore its output this frame.
                    cv2.putText(display_frame, f"OCCLUDED (Color Drop: {color_similarity:.2f})", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if not target_locked:
            # Write 0s so the dataset grader knows we didn't track the bush
            results_file.write("0,0,0,0\n")

        cv2.imshow("Color Gated Tracker", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    results_file.close()
    cv2.destroyAllWindows()
    print(f"\n✨ Done! Saved coordinates to '{RESULTS_PATH}'")

if __name__ == "__main__":
    run_color_gated_evaluation()