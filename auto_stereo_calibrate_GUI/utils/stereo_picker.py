
"""
空間鎖定版智能雙目棋盤格圖片挑選器 (Spatial Locked Stereo Selector)
Refactored for integration into Auto Stereo Calibrate GUI.
"""

import os
from pathlib import Path
from collections import defaultdict
import json
import glob
import copy
import random
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 設置 matplotlib 中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedStereoChessboardSelector:
    def __init__(self, mtxL, distL, mtxR, distR, chessboard_size=(9, 6), target_pairs=15):
        self.chessboard_size = chessboard_size
        self.target_pairs = target_pairs
        self.paired_infos = [] 
        
        # 使用傳入的相機參數作為篩選基準
        self.mtxL = mtxL
        self.distL = distL
        self.mtxR = mtxR
        self.distR = distR
        
        # 建立物件點
        self.square_size = 0.09
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

    # ---------- 基礎計算工具 ----------
    def calculate_static_reprojection_error(self, corners, mtx, dist):
        if corners is None: return float('inf')
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners, mtx, dist)
        if not ret: return float('inf')
        imgpoints2, _ = cv2.projectPoints(self.objp, rvec, tvec, mtx, dist)
        error = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        return float(error)

    def calculate_geometry_features(self, corners, image_shape):
        corners_2d = corners.reshape(-1, 2)
        center_x = float(np.mean(corners_2d[:, 0]) / image_shape[1])
        center_y = float(np.mean(corners_2d[:, 1]) / image_shape[0])
        grid_x = int(min(int(center_x * 3), 2))
        grid_y = int(min(int(center_y * 3), 2))
        region = int(grid_y * 3 + grid_x)
        
        x_range = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
        y_range = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
        coverage = float((x_range * y_range) / (image_shape[0] * image_shape[1]))
        
        first_row_vec = corners_2d[self.chessboard_size[0]-1] - corners_2d[0]
        angle_rad = np.arctan2(first_row_vec[1], first_row_vec[0])
        angle_deg = abs(np.degrees(angle_rad))
        if angle_deg > 90: angle_deg = 180 - angle_deg
        
        scale = (x_range / image_shape[1] + y_range / image_shape[0]) / 2.0
        
        return {
            'center_x': float(center_x),
            'center_y': float(center_y),
            'region': int(region),
            'coverage': float(coverage),
            'angle': float(angle_deg),
            'scale': float(scale)
        }

    def _detect_single(self, img_path: Path, is_left: bool):
        # Support Chinese paths using pure python read + imdecode
        img = None
        try:
            with open(str(img_path), 'rb') as f:
                file_bytes = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            pass

        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if not ret: return None
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        feats = self.calculate_geometry_features(refined, gray.shape)
        
        mtx = self.mtxL if is_left else self.mtxR
        dist = self.distL if is_left else self.distR
        static_error = self.calculate_static_reprojection_error(refined, mtx, dist)
        
        angle_bonus = 1.0 + (feats['angle'] / 90.0) 
        error_penalty = 1.0
        if static_error > 1.0:
            error_penalty = 1.0 / static_error
            
        final_score = (feats['coverage'] * 100) * angle_bonus * error_penalty
        
        return {
            'path': str(img_path),
            'filename': img_path.name,
            'corners': refined, 
            'score': float(final_score),
            'static_error': float(static_error),
            **feats
        }

    def analyze_and_pair(self, left_dir: str, right_dir: str, logger=print):
        logger("正在分析影像特徵與配對...")
        left_files = sorted(list(Path(left_dir).glob('*.[jJpP]*')))
        right_files = sorted(list(Path(right_dir).glob('*.[jJpP]*')))
        right_map = {p.name: p for p in right_files}
        
        paired = []
        for lf in left_files:
            rf = right_map.get(lf.name)
            if rf is None: continue
            
            l_info = self._detect_single(lf, is_left=True)
            r_info = self._detect_single(rf, is_left=False)
            
            if l_info and r_info:
                if l_info['static_error'] > 10.0 or r_info['static_error'] > 10.0:
                    continue
                
                pair_score = min(l_info['score'], r_info['score'])
                
                paired.append({
                    'pair_key': lf.name,
                    'left': l_info,
                    'right': r_info,
                    'pair_score': float(pair_score),
                    'pair_region': int(l_info['region']),
                    'static_error_sum': float(l_info['static_error'] + r_info['static_error'])
                })
        
        self.paired_infos = paired
        logger(f"原始配對數: {len(left_files)}, 有效檢測配對數: {len(paired)}")
        return len(paired)

    def perform_stereo_calibration(self, selected_pairs):
        objpoints = [self.objp for _ in selected_pairs]
        imgpoints_L = [p['left']['corners'] for p in selected_pairs]
        imgpoints_R = [p['right']['corners'] for p in selected_pairs]
        
        if not imgpoints_L: return float('inf'), None, None, None, None, None, None

        # img_size = cv2.imread(selected_pairs[0]['left']['path'], 0).shape[::-1]
        
        # Safe read for image size
        def read_img_safe(path):
            img = None
            try:
                with open(path, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            except: pass
            return img

        temp_img = read_img_safe(selected_pairs[0]['left']['path'])
        if temp_img is None: return float('inf'), None, None, None, None, None, None
        img_size = temp_img.shape[::-1]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        
        try:
            ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_L, imgpoints_R,
                self.mtxL, self.distL, self.mtxR, self.distR,
                img_size, criteria=criteria, flags=flags
            )
            return float(ret), M1, D1, M2, D2, R, T
        except cv2.error:
            return float('inf'), None, None, None, None, None, None

    def select_best_pairs(self, target_rmse=0.5, logger=print):
        if len(self.paired_infos) < self.target_pairs:
            logger("可用圖片不足目標數量，回傳全部。")
            return self.paired_infos, 0.0

        logger(f"\n=== 開始智能挑選 (目標: {self.target_pairs}對, RMS < {target_rmse}px) ===")
        
        candidates = sorted(self.paired_infos, key=lambda x: x['pair_score'], reverse=True)
        
        # 2. 初始填充 (保證 0-8 區域各一張)
        selected = []
        region_map = defaultdict(list)
        for p in candidates:
            region_map[p['pair_region']].append(p)
            
        for r in range(9):
            if region_map[r]:
                selected.append(region_map[r][0])
        
        logger(f"初始空間覆蓋選取: {len(selected)} 對")

        # 3. 多樣性填充
        needed = self.target_pairs - len(selected)
        if needed > 0:
            current_ids = {x['pair_key'] for x in selected}
            pool = [x for x in candidates if x['pair_key'] not in current_ids]
            
            if len(pool) > 0:
                feats = np.array([[x['left']['angle'], x['right']['angle'], x['left']['scale']] for x in pool])
                # 防止除以零
                std_feat = feats.std(axis=0)
                std_feat[std_feat == 0] = 1e-5
                feats = (feats - feats.mean(axis=0)) / std_feat
                
                n_clusters = min(needed, len(pool))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
                
                cluster_groups = defaultdict(list)
                for idx, label in enumerate(kmeans.labels_):
                    cluster_groups[label].append(pool[idx])
                
                for lbl in range(n_clusters):
                    if cluster_groups[lbl]:
                        best_in_cluster = max(cluster_groups[lbl], key=lambda x: x['pair_score'])
                        selected.append(best_in_cluster)
                
                while len(selected) < self.target_pairs:
                    rem_pool = [x for x in candidates if x not in selected]
                    if not rem_pool: break
                    selected.append(rem_pool[0])
            
        selected = selected[:self.target_pairs]
        
        # 4. 迭代優化 (含區域鎖定機制)
        current_rms, M1, D1, M2, D2, R, T = self.perform_stereo_calibration(selected)
        logger(f"初始組合 RMS: {current_rms:.4f} px")
        
        iteration = 0
        max_iter = 50
        
        while current_rms > target_rmse and iteration < max_iter:
            iteration += 1
            max_pair_error = -1
            worst_pair_index = -1
            
            # 使用 solvePnP 估算每張圖的誤差貢獻
            for i, p in enumerate(selected):
                retL, rvecL, tvecL = cv2.solvePnP(self.objp, p['left']['corners'], M1, D1)
                retR, rvecR, tvecR = cv2.solvePnP(self.objp, p['right']['corners'], M2, D2)
                
                if retL and retR:
                    projL, _ = cv2.projectPoints(self.objp, rvecL, tvecL, M1, D1)
                    projR, _ = cv2.projectPoints(self.objp, rvecR, tvecR, M2, D2)
                    errL = cv2.norm(p['left']['corners'], projL, cv2.NORM_L2) / np.sqrt(len(projL))
                    errR = cv2.norm(p['right']['corners'], projR, cv2.NORM_L2) / np.sqrt(len(projR))
                    avg_err = (errL + errR) / 2.0
                    
                    if avg_err > max_pair_error:
                        max_pair_error = avg_err
                        worst_pair_index = i
            
            if worst_pair_index == -1: break
            
            worst_pair = selected[worst_pair_index]
            worst_region = worst_pair['pair_region']
            
            # === [核心修正] 區域鎖定檢查 ===
            count_in_region = sum(1 for p in selected if p['pair_region'] == worst_region)
            is_sole_survivor = (count_in_region <= 1)
            
            pool = [x for x in candidates if x not in selected]
            same_region_candidates = [x for x in pool if x['pair_region'] == worst_region]
            
            replacement = None
            is_cross_region_replacement = False
            
            if is_sole_survivor:
                # 該區域只剩這一張，強制只能找同區域替補
                if same_region_candidates:
                    replacement = max(same_region_candidates, key=lambda x: x['pair_score'])
                else:
                    pass 
            else:
                # 該區域有多張圖，允許跨區替換
                if same_region_candidates:
                    replacement = max(same_region_candidates, key=lambda x: x['pair_score'])
                elif pool:
                    replacement = max(pool, key=lambda x: x['pair_score'])
                    is_cross_region_replacement = True
            
            if replacement:
                new_selection = selected.copy()
                new_selection[worst_pair_index] = replacement
                
                new_rms, nM1, nD1, nM2, nD2, nR, nT = self.perform_stereo_calibration(new_selection)
                
                if new_rms < current_rms:
                    region_msg = f"(跨區: {worst_region}->{replacement['pair_region']})" if is_cross_region_replacement else f"(同區: {worst_region})"
                    logger(f"Iter {iteration}: 替換 Index {worst_pair_index} {region_msg} Err:{max_pair_error:.3f}->OK. 新 RMS: {new_rms:.4f}")
                    selected = new_selection
                    current_rms = new_rms
                    M1, D1, M2, D2, R, T = nM1, nD1, nM2, nD2, nR, nT
                else:
                    if not is_sole_survivor and not is_cross_region_replacement and pool:
                         best_global = max(pool, key=lambda x: x['pair_score'])
                         new_selection[worst_pair_index] = best_global
                         new_rms_g, _, _, _, _, _, _ = self.perform_stereo_calibration(new_selection)
                         if new_rms_g < current_rms:
                             logger(f"Iter {iteration}: 同區失敗，跨區替換成功 -> {best_global['pair_key']}。新 RMS: {new_rms_g:.4f}")
                             selected = new_selection
                             current_rms = new_rms_g
                         else:
                             break 
                    else:
                        break
            else:
                break
                
        logger(f"最終 RMS: {current_rms:.4f} px")
        
        final_regions = set(p['pair_region'] for p in selected)
        missing_regions = set(range(9)) - final_regions
        if missing_regions:
            logger(f"警告: 最終結果缺少區域: {missing_regions} (可能是原始資料該區域無可用圖片)")
        else:
            logger("確認: 最終結果覆蓋所有 9 個區域。")
            
        return selected, current_rms

    def save_results(self, selected_pairs, final_rms, out_left_dir, out_right_dir, logger=print):
        logger("\n=== 開始輸出結果 ===")
        os.makedirs(out_left_dir, exist_ok=True)
        os.makedirs(out_right_dir, exist_ok=True)
        
        summary_data = []
        
        logger("正在儲存選定的圖片...")
        for i, p in enumerate(selected_pairs):
            try:
                idx_str = f"{i+1:02d}"
                lname = f"stereo_{idx_str}_L_{p['left']['filename']}"
                rname = f"stereo_{idx_str}_R_{p['right']['filename']}"
                
                # cv2.imwrite(os.path.join(out_left_dir, lname), cv2.imread(p['left']['path']))
                # cv2.imwrite(os.path.join(out_right_dir, rname), cv2.imread(p['right']['path']))
                
                # Support Chinese paths for read/write
                def read_img_safe(path):
                    img = None
                    try:
                        with open(path, 'rb') as f:
                            file_bytes = np.frombuffer(f.read(), np.uint8)
                            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    except: pass
                    return img

                def write_img_safe(path, img):
                    success, encoded_img = cv2.imencode('.jpg', img)
                    if success:
                        with open(path, 'wb') as f:
                            f.write(encoded_img.tobytes())
                
                imgL = read_img_safe(p['left']['path'])
                imgR = read_img_safe(p['right']['path'])
                
                if imgL is not None: write_img_safe(os.path.join(out_left_dir, lname), imgL)
                if imgR is not None: write_img_safe(os.path.join(out_right_dir, rname), imgR)
                
                summary_data.append({
                    'id': int(i+1),
                    'original_name': str(p['pair_key']),
                    'region': int(p['pair_region']),
                    'score': float(round(p['pair_score'], 2)),
                    'angle_L': float(round(p['left']['angle'], 1)),
                    'static_error': float(round(p['static_error_sum'], 3))
                })
            except Exception as e:
                logger(f"儲存圖片或準備資料時發生錯誤 (ID {i+1}): {e}")

        json_path = os.path.join(Path(out_left_dir).parent.parent, 'calibration_selection_report.json')
        try:
            report = {
                'total_pairs': int(len(selected_pairs)),
                'final_rms': float(round(final_rms, 4)),
                'success_target': bool(final_rms < 0.5),
                'details': summary_data
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger(f"JSON 報表已輸出至: {json_path}")
        except Exception as e:
            logger(f"\n[警告] JSON 報表儲存失敗: {e}")
            traceback.print_exc()
        
        try:
            self.plot_analysis(selected_pairs, final_rms, Path(out_left_dir).parent.parent)
        except Exception as e:
            logger(f"[警告] 繪圖分析失敗: {e}")

    def plot_analysis(self, pairs, rms, out_dir):
        regions = [p['pair_region'] for p in pairs]
        angles = [p['left']['angle'] for p in pairs]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        region_counts = np.zeros(9)
        for r in regions: region_counts[r] += 1
        
        axes[0].bar(range(9), region_counts, color='skyblue', edgecolor='black')
        axes[0].set_title('選取圖片空間分佈 (3x3 Grid)')
        axes[0].set_xlabel('區域 ID (0-8)')
        axes[0].set_ylabel('圖片數量')
        axes[0].set_xticks(range(9))
        
        axes[1].hist(angles, bins=10, color='salmon', edgecolor='black', alpha=0.7)
        axes[1].set_title(f'圖片傾斜角度分佈 (RMS: {rms:.3f}px)')
        axes[1].set_xlabel('角度 (度)')
        axes[1].set_ylabel('頻率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'selection_analysis.png'))


def run_stereo_pick(left_dir, right_dir, out_left, out_right, mtxL, distL, mtxR, distR, chessboard_size=(9, 6), target_pairs=15, logger=print):
    selector = OptimizedStereoChessboardSelector(mtxL, distL, mtxR, distR, chessboard_size, target_pairs)
    
    count = selector.analyze_and_pair(left_dir, right_dir, logger)
    if count == 0:
        logger("未發現有效圖片，程式結束。")
        return 0, 0.0

    selected, final_rms = selector.select_best_pairs(target_rmse=0.5, logger=logger)
    selector.save_results(selected, final_rms, out_left, out_right, logger)
    
    return len(selected), final_rms
