import cv2
import os
import shutil
from pathlib import Path

def extract_frames(input_video_path, output_folder, interval=5, progress_callback=None):
    """
    Step 1: 從影片每隔指定間隔提取幀並儲存到 output_folder (origin_image)。
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片：{input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            # 使用更具描述性的命名，例如 frame_000000.jpg
            filename = f"frame_{frame_count:06d}.jpg"
            output_path = os.path.join(output_folder, filename)
            
            success, encoded_img = cv2.imencode('.jpg', frame)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                saved_count += 1
        
        frame_count += 1
        
        if progress_callback and frame_count % 100 == 0:
             progress_callback(f"正在提取幀... ({frame_count}/{total_frames})")

    cap.release()
    return saved_count

def filter_valid_images(input_folder, output_folder, chessboard_size=(9, 6), progress_callback=None):
    """
    (Optional Step): 檢查 input_folder 中的圖片是否包含完整的棋盤格。
    雖然使用者流程沒有明確要求這一步獨立出來 (由 auto_pick 處理)，
    但保留此功能以備不時之需。
    """
    os.makedirs(output_folder, exist_ok=True)
    
    images = list(Path(input_folder).glob("*.jpg"))
    total_imgs = len(images)
    valid_count = 0
    
    for idx, img_path in enumerate(images):
        img_path_str = str(img_path)
        img = None
        try:
            with open(img_path_str, 'rb') as f:
                import numpy as np
                img_data = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception:
            pass
        
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            filename = img_path.name
            dest_path = os.path.join(output_folder, filename)
            shutil.copy2(img_path_str, dest_path)
            valid_count += 1
            
        if progress_callback and idx % 10 == 0:
            progress_callback(f"正在篩選圖片... ({idx+1}/{total_imgs}) - 有效: {valid_count}")
            
    return valid_count
