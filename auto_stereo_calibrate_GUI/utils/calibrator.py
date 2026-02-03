import numpy as np
import cv2
import glob
import os
from pathlib import Path

def stereo_calibration(
    left_images_dir,
    right_images_dir,
    mtxL, distL, mtxR, distR,
    save_path, # stereo_rt.npz
    chessboard_size=(9, 6),
    square_size=0.09,
    logger=print
):
    """
    Step 5 & 6: 對 final_image 資料夾中的圖片進行雙目標定，並輸出結果。
    參考: User Code Snippet @ 3Dprojection.ipynb
    """
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    # 建立棋盤格世界座標點（ Z=0 ）
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 讀取左右相機同一幀下的棋盤格圖片
    left_images = sorted(glob.glob(os.path.join(left_images_dir, '*.jpg')))
    right_images = sorted(glob.glob(os.path.join(right_images_dir, '*.jpg')))

    if len(left_images) != len(right_images):
        logger(f"[警告] 左右路徑圖片數量不一致: Left={len(left_images)}, Right={len(right_images)}")
        # 嘗試只處理配對成功的
        common_names = set(os.path.basename(p) for p in left_images) & set(os.path.basename(p) for p in right_images)
        # But wait, logic assumes 'sorted' matches because filenames are like stereo_01_L_... and stereo_01_R_...
        # So we should match by prefix ID or index if they were saved sequentially by picker.
        # Actually, picker saves stereo_XX_L and stereo_XX_R.
        # So sorted() should align them perfectly by ID.
        pass

    imgpoints_left = []  # 左相機圖片2D點
    imgpoints_right = [] # 右相機圖片2D點
    objpoints = []       # 棋盤格世界座標點

    image_size = None
    
    logger(f"開始檢測棋盤格 (共 {len(left_images)} 組)...")

    for frameL, frameR in zip(left_images, right_images):
        
        # Support Chinese paths
        def read_img_safe(path):
            img = None
            try:
                with open(path, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                pass
            return img

        img_left = read_img_safe(frameL)
        img_right = read_img_safe(frameR)
        
        if img_left is None or img_right is None:
            continue

        grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = grayL.shape[::-1]

        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

    if not objpoints:
        logger("[錯誤] 無法在圖片中檢測到棋盤格")
        return None

    # 固定內參，僅估計 R, T 外參
    # flags = cv2.CALIB_FIX_INTRINSIC
    flags = cv2.CALIB_USE_INTRINSIC_GUESS # 使用者指定 Use Intrinsic Guess
    
    logger("執行 stereoCalibrate ...")
    ret, mtxL_opt, distL_opt, mtxR_opt, distR_opt, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR,
        image_size, criteria=criteria, flags=flags)
    
    logger(f"image_size: {image_size}")
    logger(f"RMS (ret): {ret}")
    
    # Calculate Baseline
    baseline = np.linalg.norm(T)
    logger(f"Baseline: {baseline} meters")

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, 
             mtxL_opt=mtxL_opt, distL_opt=distL_opt, 
             mtxR_opt=mtxR_opt, distR_opt=distR_opt, 
             R=R, T=T, ret=ret, baseline=baseline)
             
    logger(f"結果已儲存至: {save_path}")
    
    return {
        'R': R,
        'T': T,
        'baseline': baseline,
        'ret': ret,
        'mtxL_opt': mtxL_opt,
        'distL_opt': distL_opt,
        'mtxR_opt': mtxR_opt,
        'distR_opt': distR_opt
    }
