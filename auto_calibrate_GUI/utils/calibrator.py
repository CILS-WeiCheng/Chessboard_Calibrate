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
    執行單目相機最終標定：讀取選定影像 -> 執行標定 -> 計算誤差 -> 儲存結果 (.npz & .txt)
    """
    logger(f"=== 開始最終標定 ===")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    obj_points, img_points = [], []
    images = glob.glob(image_folder_pattern)
    if not images:
        logger("[錯誤] 找不到圖片進行標定")
        return None
    
    img_shape = None
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            try:
                with open(fname, 'rb') as f:
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            except Exception: continue
        
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            obj_points.append(objp)
            img_points.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            
    if not obj_points:
        logger("[錯誤] 在這些圖片中無法檢測到棋盤格")
        return None

    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None,
            criteria=criteria, flags=cv2.CALIB_FIX_K3
        )
    except Exception as e:
        logger(f"[錯誤] 標定失敗: {e}"); return None

    # 誤差分析與驗證
    per_view_errors = []
    for i in range(len(obj_points)):
        p2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(img_points[i], p2, cv2.NORM_L2) / len(p2)
        per_view_errors.append(err)

    mean_err = np.mean(per_view_errors)
    
    # ── 參數合理性驗證 ──────────────────────────────────────────────────────────
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    img_w, img_h = img_shape

    # 驗證1：焦距一致性（正方形像素的相機 fx 與 fy 應小於 1%）
    focal_diff_pct = abs(fx - fy) / min(fx, fy) * 100
    if focal_diff_pct > 2.0:
        logger(f"\u26a0️  [焦距警告] fx ({fx:.1f}) 與 fy ({fy:.1f}) 差異 {focal_diff_pct:.2f}%，"
               f"超過 2%！詳診：標定圖片可能缺乏足夠傾斜角，"
               f"焦距检測存在數學模糊性。建議重新拍攝包含大傾斜角的標定圖片後再標定。")
    else:
        logger(f"✅ [焦距驗證通過] fx ({fx:.1f}) 與 fy ({fy:.1f}) 差異 {focal_diff_pct:.2f}%，符合正方形像素標準。")

    # 驗證2：主點偽移量（應要跟團片寬高的一半接近）
    cx_expected, cy_expected = img_w / 2.0, img_h / 2.0
    cx_dev = abs(cx - cx_expected)
    cy_dev = abs(cy - cy_expected)
    if cx_dev > 30 or cy_dev > 30:
        logger(f"\u26a0️  [主點警告] cx={cx:.1f} (期望~{cx_expected:.0f}, 偏移 {cx_dev:.1f}px), "
               f"cy={cy:.1f} (期望~{cy_expected:.0f}, 偏移 {cy_dev:.1f}px)。"
               f"偏移超過 30 像素！標定圖片可能未覆蓋畫面邊緣，導致畜變系數與主點無法正確分離。")
    else:
        logger(f"✅ [主點驗證通過] cx={cx:.1f}, cy={cy:.1f} （偏移分別為 {cx_dev:.1f}px, {cy_dev:.1f}px）。")
    
    validation_results = {
        'mean_reprojection_error': mean_err,
        'max_reprojection_error': max(per_view_errors),
        'min_reprojection_error': min(per_view_errors),
        'std_reprojection_error': np.std(per_view_errors),
        'per_view_errors': per_view_errors,
        'focal_lengths': (fx, fy),
        'focal_diff_pct': focal_diff_pct,
        'principal_point': (cx, cy),
        'cx_deviation_px': cx_dev,
        'cy_deviation_px': cy_dev,
        'distortion_coeffs': dist[0].tolist()
    }

    # 儲存結果 (.npz & .txt)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs, validation_results=validation_results)
    
    txt_path = str(Path(save_path).with_suffix('.txt'))
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Mean Reprojection Error: {mean_err}\n")
        f.write(f"Camera Matrix:\n{mtx}\n")
        f.write(f"Distortion Coefficients:\n{dist}\n")
        
    logger(f"標定完成。平均誤差: {mean_err:.6f}\n結果已儲存至: {save_path}")
    return validation_results
