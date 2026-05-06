import cv2
import os
import shutil
import numpy as np
from pathlib import Path

def extract_frames(input_video_path, output_folder, interval=5, progress_callback=None):
    """從影片每隔指定間隔提取幀並儲存為 .jpg"""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened(): raise ValueError(f"無法開啟影片：{input_video_path}")

    total, count, saved = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % interval == 0:
            out_path = os.path.join(output_folder, f"frame_{count:06d}.jpg")
            _, enc = cv2.imencode('.jpg', frame)
            with open(out_path, 'wb') as f: f.write(enc.tobytes())
            saved += 1
        count += 1
        if progress_callback and count % 100 == 0: progress_callback(f"提取進度: {count}/{total}")
    cap.release(); return saved

def filter_valid_images(input_folder, output_folder, chessboard_size=(9, 6), progress_callback=None):
    """篩選資料夾中包含完整棋盤格的圖片並複製到輸出目錄"""
    os.makedirs(output_folder, exist_ok=True)
    imgs = list(Path(input_folder).glob("*.jpg"))
    valid = 0
    for i, p in enumerate(imgs):
        try:
            with open(str(p), 'rb') as f: img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None and cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), chessboard_size, None)[0]:
                shutil.copy2(str(p), os.path.join(output_folder, p.name)); valid += 1
        except Exception: pass
        if progress_callback and i % 10 == 0: progress_callback(f"篩選進度: {i+1}/{len(imgs)} - 有效: {valid}")
    return valid
