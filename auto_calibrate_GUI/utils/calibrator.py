import numpy as np
import cv2
import glob
import os
from pathlib import Path

def calibration_final(
    image_folder_pattern, 
    save_path, # .npz path
    chessboard_size=(9, 6), 
    square_size=0.09,
    logger=print
):
    """
    Step 5 & 6: Perform final calibration on selected images and save results.
    """
    logger(f"=== 開始最終標定 ===")
    logger(f"讀取圖片路徑: {image_folder_pattern}")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points = []
    
    images = glob.glob(image_folder_pattern)
    if not images:
        logger("[錯誤] 找不到圖片進行標定")
        return None
    
    img_shape = None
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            # Handle Chinese paths / non-ASCII paths
            try:
                with open(fname, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                pass
        
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            
    if not obj_points:
        logger("[錯誤] 在這些圖片中無法檢測到棋盤格")
        return None

    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )
    except Exception as e:
        logger(f"[錯誤] 標定失敗: {e}")
        return None

    total_error = 0
    per_view_errors = []
    for i in range(len(obj_points)):
        projected_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
        per_view_errors.append(error)
        total_error += error

    mean_reprojection_error = total_error / len(obj_points)
    
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # Validation results dict similar to user's code
    validation_results = {
        'mean_reprojection_error': mean_reprojection_error,
        'max_reprojection_error': max(per_view_errors),
        'min_reprojection_error': min(per_view_errors),
        'std_reprojection_error': np.std(per_view_errors),
        'per_view_errors': per_view_errors,
        'focal_lengths': (fx, fy),
        'principal_point': (cx, cy),
        'distortion_coeffs': dist_coeffs[0].tolist()
    }

    # Save NPZ
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs, 
             rvecs=rvecs,
             tvecs=tvecs,
             validation_results=validation_results)
             
    logger(f"標定完成。平均重投影誤差: {mean_reprojection_error:.6f}")
    logger(f"結果已儲存至: {save_path}")
    
    # Save text report
    txt_path = str(Path(save_path).with_suffix('.txt'))
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Mean Reprojection Error: {mean_reprojection_error}\n")
        f.write(f"Camera Matrix:\n{camera_matrix}\n")
        f.write(f"Distortion Coefficients:\n{dist_coeffs}\n")
        
    return validation_results
