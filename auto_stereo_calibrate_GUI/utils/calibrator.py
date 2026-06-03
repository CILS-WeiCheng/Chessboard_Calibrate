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
    執行雙目相機最終標定：讀取選定配對影像 -> 執行標定 -> 輸出 R, T 與 Baseline 並儲存 .npz。
    """
    logger(f"=== 開始雙目最終標定 ===")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # 讀取並配對影像
    l_imgs = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        l_imgs.extend(glob.glob(os.path.join(left_images_dir, ext)))
    l_imgs = sorted(l_imgs)
    
    obj_pts, img_ptsL, img_ptsR = [], [], []
    img_size = None
    
    for fl in l_imgs:
        fr = os.path.join(right_images_dir, os.path.basename(fl).replace('_L_', '_R_'))
        if not os.path.exists(fr): continue
        
        try:
            with open(fl, 'rb') as f: il = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            with open(fr, 'rb') as f: ir = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception: continue
        
        if il is None or ir is None: continue
        grayL, grayR = cv2.cvtColor(il, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        if img_size is None: img_size = grayL.shape[::-1]

        retL, cL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cR = cv2.findChessboardCorners(grayR, chessboard_size, None)
        if retL and retR:
            obj_pts.append(objp)
            img_ptsL.append(cv2.cornerSubPix(grayL, cL, (11, 11), (-1, -1), criteria))
            img_ptsR.append(cv2.cornerSubPix(grayR, cR, (11, 11), (-1, -1), criteria))

    if not obj_pts:
        logger("[錯誤] 無法在圖片中檢測到棋盤格"); return None

    logger("執行 stereoCalibrate ...")
    ret, mtxL_o, distL_o, mtxR_o, distR_o, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_ptsL, img_ptsR, mtxL, distL, mtxR, distR,
        img_size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    
    baseline = np.linalg.norm(T)
    
    # ── 極線誤差驗證（Epipolar Geometry Quality Check）───────────────────────
    # 極線誤差為校正前左右對應角點的 Y 座標差（理想值应要為 0）
    # 是評估雙目外參（R, T）品質最直接的指標，比 RMS 更能反映三角測量的潛在精度
    # 標準：< 0.5 像素（優良），< 0.3 像素（很優）
    epi_errors = []
    for ptsL, ptsR in zip(img_ptsL, img_ptsR):
        for ptL, ptR in zip(ptsL, ptsR):
            # Y 座標差即為極線偏差
            epi_err = abs(float(ptL[0][1]) - float(ptR[0][1]))
            epi_errors.append(epi_err)
    
    mean_epi_err = float(np.mean(epi_errors)) if epi_errors else float('inf')
    max_epi_err = float(np.max(epi_errors)) if epi_errors else float('inf')
    
    if mean_epi_err < 0.3:
        logger(f"✅ [極線誤差很優] 平均: {mean_epi_err:.4f} px, 最大: {max_epi_err:.4f} px（標準: < 0.3 px）")
    elif mean_epi_err < 0.5:
        logger(f"✅ [極線誤差優良] 平均: {mean_epi_err:.4f} px, 最大: {max_epi_err:.4f} px（標準: < 0.5 px）")
    else:
        logger(f"⚠️  [極線誤差警告] 平均: {mean_epi_err:.4f} px, 最大: {max_epi_err:.4f} px（超過 0.5 px！）"
               f"外參 R, T 計算危險，將導致三角測量射線無法相交，建議重新標定。")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        mtxL_opt=mtxL_o, distL_opt=distL_o,
        mtxR_opt=mtxR_o, distR_opt=distR_o,
        R=R, T=T, ret=ret, baseline=baseline,
        mean_epipolar_error=mean_epi_err,
        max_epipolar_error=max_epi_err
    )
             
    logger(f"標定完成。RMS: {ret:.6f}, Baseline: {baseline:.4f}m\n結果儲存至: {save_path}")
    return {
        'R': R, 'T': T, 'baseline': baseline, 'ret': ret,
        'mtxL_opt': mtxL_o, 'distL_opt': distL_o,
        'mtxR_opt': mtxR_o, 'distR_opt': distR_o,
        'mean_epipolar_error': mean_epi_err,
        'max_epipolar_error': max_epi_err
    }
