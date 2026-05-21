import cv2
import numpy as np
import os
import sys

# --- CONFIGURATION ---
VIDEO_PATH = 'inputs/samp1.mp4'
IMG_PATH = 'inputs/img1.png'
OUTPUT_PATH = 'outputs/final_ultra_scale.mp4'

# --- ULTRA SETTINGS ---
# 0.05 = 5% size. We scan 30 steps to be very precise.
SCAN_SCALES = np.linspace(0.15, 1.0, 30) 

# STRICT RULES (To prevent blurring random trees)
MATCH_THRESHOLD = 0.65  # slightly lower than 0.65 to catch tiny objects, but strict enough to avoid noise
MAX_JUMP_DIST = 150     # Allowed movement per frame (pixels)
MIN_PIXEL_SIZE = 12     # Don't match anything smaller than 12x12 pixels

def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def run_ultra_tracker():
    print("🚀 Launching Ultra-Scale Tracker...")
    
    # 1. Load Reference
    template = cv2.imread(IMG_PATH)
    if template is None: print("❌ Error: Image not found"); return
    
    # Sharpen template to define the camera edges clearly
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    template = cv2.filter2D(template, -1, kernel)
    
    th, tw = template.shape[:2]
    
    # 2. Setup Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not os.path.exists('outputs'): os.makedirs('outputs')
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # State Variables
    last_center = None
    last_box = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        sys.stdout.write(f"\rProcessing {frame_count}/{total_frames}")
        sys.stdout.flush()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        best_val = -1
        best_rect = None
        
        # 3. ULTRA-WIDE SEARCH
        for scale in SCAN_SCALES:
            nw, nh = int(tw * scale), int(th * scale)
            
            # GUARDRAIL: Stop if it gets too small (noise) or too big
            if nw < MIN_PIXEL_SIZE or nh < MIN_PIXEL_SIZE or nw > width or nh > height: 
                continue
            
            resized_tmpl = cv2.resize(gray_template, (nw, nh))
            res = cv2.matchTemplate(gray_frame, resized_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_rect = (max_loc[0], max_loc[1], nw, nh)

        # 4. LOGIC GATES
        valid_match = False
        
        if best_val > MATCH_THRESHOLD:
            x, y, w, h = best_rect
            cx, cy = x + w//2, y + h//2
            
            if last_center is None:
                # First lock
                valid_match = True
            else:
                # Check distance (Did it teleport?)
                move_dist = dist((cx, cy), last_center)
                if move_dist < MAX_JUMP_DIST:
                    valid_match = True
                else:
                    valid_match = False 

        # 5. RENDER
        display_frame = frame.copy()
        
        if valid_match:
            x, y, w, h = best_rect
            last_box = (x, y, w, h)
            last_center = (x + w//2, y + h//2)
            
            # --- BLUR ONLY THE CAMERA ---
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x+w), min(height, y+h)
            
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
            frame[y1:y2, x1:x2] = blurred_roi
            
            # HUD
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"OBJ ({best_val:.2f})", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        else:
            # If lost, reset center so it can search globally again next frame
            # (But don't blur anything)
            if last_box:
                 x, y, w, h = last_box
                 cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                 cv2.putText(display_frame, "LOST", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Only clear history if score is VERY low, otherwise keep memory for a bit
            if best_val < 0.3:
                last_center = None

        out.write(frame)
        cv2.imshow("Ultra Scale Tracker", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✨ Ultra-Scale tracking complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_ultra_tracker()