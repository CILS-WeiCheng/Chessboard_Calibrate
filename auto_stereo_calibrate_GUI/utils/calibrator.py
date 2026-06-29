import numpy as np
import cv2
import glob
import os
from pathlib import Path

def align_corners(ptsL, ptsR, mtxL, distL, mtxR, distR, img_size, chessboard_size=(9, 6), square_size=0.091):
    """
    自動評估 4 種角點旋轉/翻轉可能性，並返回極線誤差最小的對齊後右相機角點。
    """
    cR_orig = ptsR.copy()
    cR_rev = ptsR[::-1].copy()
    cols, rows = chessboard_size
    cR_h = ptsR.reshape(rows, cols, 2)[:, ::-1, :].reshape(-1, 1, 2)
    cR_v = ptsR.reshape(rows, cols, 2)[::-1, :, :].reshape(-1, 1, 2)

    cases = [
        ("original", cR_orig),
        ("reversed", cR_rev),
        ("horizontal", cR_h),
        ("vertical", cR_v)
    ]

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    best_case = "original"
    best_corners = cR_orig
    min_epi_err = float('inf')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    for name, corners_R in cases:
        try:
            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                [objp], [ptsL], [corners_R],
                mtxL, distL, mtxR, distR,
                img_size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
            )
            ptsL_flat = ptsL.reshape(-1, 2)
            ptsR_flat = corners_R.reshape(-1, 2)
            linesR = cv2.computeCorrespondEpilines(ptsL_flat, 1, F).reshape(-1, 3)
            errors = []
            for ptR, lR in zip(ptsR_flat, linesR):
                d = abs(lR[0]*ptR[0] + lR[1]*ptR[1] + lR[2]) / np.sqrt(lR[0]**2 + lR[1]**2)
                errors.append(d)
            mean_err = np.mean(errors)

            if mean_err < min_epi_err:
                min_epi_err = mean_err
                best_case = name
                best_corners = corners_R
        except Exception:
            continue

    return best_corners, best_case, min_epi_err

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
    
    from collections import defaultdict
    candidates = []
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
            refL = cv2.cornerSubPix(grayL, cL, (11, 11), (-1, -1), criteria)
            refR = cv2.cornerSubPix(grayR, cR, (11, 11), (-1, -1), criteria)
            
            cR_orig = refR.copy()
            cR_rev = refR[::-1].copy()
            cols, rows = chessboard_size
            cR_h = refR.reshape(rows, cols, 2)[:, ::-1, :].reshape(-1, 1, 2)
            cR_v = refR.reshape(rows, cols, 2)[::-1, :, :].reshape(-1, 1, 2)
            
            cases = {
                "original": cR_orig,
                "reversed": cR_rev,
                "horizontal": cR_h,
                "vertical": cR_v
            }
            
            errors = {}
            for name, corners_R in cases.items():
                try:
                    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                        [objp], [refL], [corners_R],
                        mtxL, distL, mtxR, distR,
                        img_size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
                    )
                    ptsL_flat = refL.reshape(-1, 2)
                    ptsR_flat = corners_R.reshape(-1, 2)
                    linesR = cv2.computeCorrespondEpilines(ptsL_flat, 1, F).reshape(-1, 3)
                    errs = []
                    for ptR, lR in zip(ptsR_flat, linesR):
                        d = abs(lR[0]*ptR[0] + lR[1]*ptR[1] + lR[2]) / np.sqrt(lR[0]**2 + lR[1]**2)
                        errs.append(d)
                    errors[name] = np.mean(errs)
                except Exception:
                    errors[name] = 9999.0
                    
            valid_errs = [v for v in errors.values() if v < 9000.0]
            if len(valid_errs) >= 2:
                delta = max(valid_errs) - min(valid_errs)
                best_case = min(errors, key=errors.get)
                candidates.append({
                    'refL': refL,
                    'cases': cases,
                    'delta': delta,
                    'best_case': best_case,
                    'errors': errors
                })

    if not candidates:
        logger("[錯誤] 無法在任何圖片對中檢測到成對棋盤格"); return None

    # 進行多數決投票
    candidates_sorted = sorted(candidates, key=lambda x: x['delta'], reverse=True)
    vote_subset = candidates_sorted[:min(20, len(candidates_sorted))]
    votes = defaultdict(int)
    for c in vote_subset:
        votes[c['best_case']] += 1
    
    session_align_mode = max(votes, key=votes.get)
    logger(f"最終標定階段投票結果（最具傾斜度前 {len(vote_subset)} 張）：{dict(votes)}")
    logger(f"➔ 決議最終標定角點對齊模式為: 【{session_align_mode}】")

    # 第二階段：收集套用此對齊模式後的角點點集
    obj_pts, img_ptsL, img_ptsR = [], [], []
    for c in candidates:
        epi_err = c['errors'][session_align_mode]
        # 排除偏離該模式大於 15.0 px 的嚴重噪訊對
        if epi_err > 15.0:
            continue
            
        obj_pts.append(objp)
        img_ptsL.append(c['refL'])
        img_ptsR.append(c['cases'][session_align_mode])

    if not obj_pts:
        logger("[錯誤] 無法在圖片中檢測到棋盤格"); return None

    logger("執行 stereoCalibrate ...")
    ret, mtxL_o, distL_o, mtxR_o, distR_o, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_ptsL, img_ptsR, mtxL, distL, mtxR, distR,
        img_size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    
    baseline = np.linalg.norm(T)
    
    # ── 極線誤差驗證（Epipolar Geometry Quality Check）───────────────────────
    # 真正的極線誤差為標定點到對應極線（由 Fundamental Matrix 算得）的距離。
    # 是評估雙目外參（R, T）品質最直接的指標，比 RMS 更能反映三角測量的潛在精度
    # 標準：< 0.5 像素（優良），< 0.3 像素（很優）
    epi_errors = []
    for ptsL, ptsR in zip(img_ptsL, img_ptsR):
        pts1 = ptsL.reshape(-1, 2)
        pts2 = ptsR.reshape(-1, 2)
        
        # 針對左圖的點，計算其在右圖上的極線
        lines2 = cv2.computeCorrespondEpilines(pts1, 1, F).reshape(-1, 3)
        for pt2, l2 in zip(pts2, lines2):
            # 計算點 pt2 到線 l2 (ax + by + c = 0) 的幾何距離
            dist_val = abs(l2[0]*pt2[0] + l2[1]*pt2[1] + l2[2]) / np.sqrt(l2[0]**2 + l2[1]**2)
            epi_errors.append(dist_val)
    
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
