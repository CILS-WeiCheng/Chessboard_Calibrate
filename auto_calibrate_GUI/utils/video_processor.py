import cv2
import os
import shutil
import numpy as np
from pathlib import Path

def extract_frames(input_video_path, output_folder, interval=5, progress_callback=None):
    """
    從影片中每隔指定間隔提取幀，並以高品質 JPG 格式儲存。
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened(): raise ValueError(f"無法開啟影片：{input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count, saved_count = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % interval == 0:
            output_path = os.path.join(output_folder, f"VUE_{frame_count:06d}.jpg")
            success, encoded_img = cv2.imencode('.jpg', frame)
            if success:
                with open(output_path, 'wb') as f: f.write(encoded_img.tobytes())
                saved_count += 1
        
        frame_count += 1
        if progress_callback and frame_count % 100 == 0:
             progress_callback(f"正在提取幀... ({frame_count}/{total_frames})")

    cap.release()
    return saved_count

def filter_valid_images(input_folder, output_folder, chessboard_size=(9, 6), progress_callback=None):
    """
    初步篩選：找出包含完整棋盤格的圖片並複製到目標資料夾。
    """
    os.makedirs(output_folder, exist_ok=True)
    images = list(Path(input_folder).glob("*.jpg"))
    valid_count = 0
    
    for idx, img_path in enumerate(images):
        img_path_str = str(img_path)
        img = cv2.imread(img_path_str)
        if img is None:
            try:
                with open(img_path_str, 'rb') as f:
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            except Exception: continue
        
        if img is None: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            shutil.copy2(img_path_str, os.path.join(output_folder, img_path.name))
            valid_count += 1
            
        if progress_callback and idx % 10 == 0:
            progress_callback(f"正在篩選圖片... ({idx+1}/{len(images)}) - 有效: {valid_count}")
            
    return valid_count
