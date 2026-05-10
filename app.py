import os
import cv2
import numpy as np
import shutil
from moviepy.editor import ImageSequenceClip


VIDEO_PATH = "inputs/samp1.mp4"        
IMG_PATH = "inputs/img1.png"           
OUTPUT_VIDEO = "outputs/Redacted_Video.mp4"


feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

SEARCH_THRESHOLD = 0.55  
MIN_POINTS = 5           

def search_for_object(frame, ref_img):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    th, tw = gray_ref.shape[:2]
    
    best_val = -1
    best_box = None
    
    for scale in np.linspace(0.1, 1.0, 20):
        nw, nh = int(tw * scale), int(th * scale)
        if nw < 10 or nh < 10 or nw > gray_frame.shape[1] or nh > gray_frame.shape[0]: 
            continue
            
        resized_ref = cv2.resize(gray_ref, (nw, nh))
        res = cv2.matchTemplate(gray_frame, resized_ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_val:
            best_val = max_val
            best_box = (max_loc[0], max_loc[1], nw, nh)
            
    if best_val > SEARCH_THRESHOLD:
        return best_box
    return None

def get_features_in_box(gray_frame, box):
    x, y, w, h = box
    mask = np.zeros_like(gray_frame)
    mask[y:y+h, x:x+w] = 255
    points = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
    return points

def main():
    print("🚀 Launching Clean Production Tracker...")
    
    temp_dir = "temp_clean_frames"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ref_img = cv2.imread(IMG_PATH)
    if ref_img is None:
        print("❌ Error: Reference image not found.")
        return

    ret, old_frame = cap.read()
    if not ret: return
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    tracking_mode = False
    p0 = None
    
    output_images = []
    count = 0
    
    print("🎥 Processing Video in the background...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        
    
        # STATE 1: SEARCH MODE
  
        if not tracking_mode:
            box = search_for_object(frame, ref_img)
            
            if box:
                p0 = get_features_in_box(frame_gray, box)
                if p0 is not None and len(p0) >= MIN_POINTS:
                    tracking_mode = True
                
     
        # STATE 2: TRACKING MODE
\
        if tracking_mode and p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            p0_reverse, st_rev, err_rev = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)
            
            if p1 is not None and p0_reverse is not None:
                d = abs(p0 - p0_reverse).reshape(-1, 2).max(-1)
                good_status = (d < 1) & (st.reshape(-1) == 1)
                
                good_new = p1[good_status]
                
                if len(good_new) >= MIN_POINTS:
                    # Target is valid. Flatten array and calculate bounding box.
                    good_new_flat = good_new.reshape(-1, 2)
                    x_min, y_min = np.int32(np.min(good_new_flat, axis=0))
                    x_max, y_max = np.int32(np.max(good_new_flat, axis=0))
                    
                    pad = 10
                    x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
                    x2, y2 = min(frame.shape[1], x_max + pad), min(frame.shape[0], y_max + pad)
                    
                    # Apply Blur (NO visual debug elements drawn)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                        display_frame[y1:y2, x1:x2] = blurred_roi
                        
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    tracking_mode = False
            else:
                tracking_mode = False

        old_gray = frame_gray.copy()
        
        out_path = os.path.join(temp_dir, f"{count:05d}.jpg")
        cv2.imwrite(out_path, display_frame)
        output_images.append(out_path)
        count += 1
        
        # Print progress to terminal instead of drawing on video
        if count % 30 == 0: 
            print(f"Processed {count} frames...")

    cap.release()
    
    print("🎬 Stitching final clean video...")
    clip = ImageSequenceClip(output_images, fps=fps)
    clip.write_videofile(OUTPUT_VIDEO, codec="libx264", logger=None)
    
    shutil.rmtree(temp_dir)
    print(f"✅ Success! Clean video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()