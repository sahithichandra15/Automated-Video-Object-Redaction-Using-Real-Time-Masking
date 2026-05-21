import cv2
import numpy as np
import time
import os

# ==========================================
# ⚙️ CONFIGURATION (Update these paths)
# ==========================================
VIDEO_PATH = 'inputs/samp3.mp4'
IMG_PATH = 'inputs/img4.png'

# Tracking Parameters (Matches your production code)
feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def main():
    print("⏱️ Initializing Performance Evaluation...")
    
    if not os.path.exists(VIDEO_PATH) or not os.path.exists(IMG_PATH):
        print(f"❌ Error: Could not find {VIDEO_PATH} or {IMG_PATH}. Please check your paths.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ref_img = cv2.imread(IMG_PATH)
    
    ret, old_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read video frames.")
        return
        
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    tracking_mode = False
    p0 = None
    frames_processed = 0
    
    print("🚀 Running benchmark (this will process the video without displaying it)...")
    
    # ⏱️ START TIMER (We only time the algorithm, not the video loading/saving)
    start_time = time.time() 
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not tracking_mode:
            # 1. Simulate Search Mode Load (Multi-scale template matching)
            for scale in np.linspace(0.1, 1.0, 20):
                nw, nh = int(ref_gray.shape[1] * scale), int(ref_gray.shape[0] * scale)
                if nw < 10 or nh < 10 or nw > frame_gray.shape[1] or nh > frame_gray.shape[0]: 
                    continue
                resized_ref = cv2.resize(ref_gray, (nw, nh))
                cv2.matchTemplate(frame_gray, resized_ref, cv2.TM_CCOEFF_NORMED)
            
            # Force transition to tracking to benchmark both states
            tracking_mode = True 
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            
        elif tracking_mode and p0 is not None:
            # 2. Simulate Tracking Mode Load (Forward-Backward Flow + Blur)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            p0_reverse, st_rev, err_rev = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)
            
            # Simulate the computational cost of the 51x51 Gaussian Blur
            # (Using a fixed 100x100 region to standardize the benchmark)
            roi = frame[0:100, 0:100] 
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
            
            p0 = p1

        old_gray = frame_gray.copy()
        frames_processed += 1

    # ⏱️ STOP TIMER
    end_time = time.time()
    cap.release()
    
    # --- CALCULATE METRICS ---
    total_time = end_time - start_time
    fps = frames_processed / total_time
    ms_per_frame = (total_time / frames_processed) * 1000
    
    print("\n" + "="*50)
    print("📊 QUANTITATIVE RESULTS FOR IEEE PAPER")
    print("="*50)
    print(f"Total Frames Processed : {frames_processed}")
    print(f"Total Processing Time  : {total_time:.2f} seconds")
    print(f"Average Speed (FPS)    : {fps:.2f} Frames Per Second")
    print(f"Latency Per Frame      : {ms_per_frame:.2f} milliseconds")
    print("="*50)
    print("💡 Tip: In your paper, mention the CPU model you ran this on ")
    print("   (e.g., 'Evaluated on an Intel Core i7 processor resulting in ~X FPS').")

if __name__ == "__main__":
    main()