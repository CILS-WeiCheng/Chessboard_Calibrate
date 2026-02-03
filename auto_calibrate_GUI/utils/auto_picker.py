import cv2
import numpy as np
import os
import shutil
import json
import traceback
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set matplotlib backend to Agg to avoid GUI issues in threads
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

class OptimizedSingleChessboardSelector:
    def __init__(self, chessboard_size=(9, 6), square_size=0.09, target_count=15, logger=print):
        self.chessboard_size = chessboard_size
        self.target_count = target_count
        self.square_size = square_size
        self.logger = logger
        self.candidates = []
        
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

    def calculate_geometry_features(self, corners, image_shape):
        corners_2d = corners.reshape(-1, 2)
        h, w = image_shape[:2]
        
        center_x = float(np.mean(corners_2d[:, 0]) / w)
        center_y = float(np.mean(corners_2d[:, 1]) / h)
        grid_x = int(min(int(center_x * 3), 2))
        grid_y = int(min(int(center_y * 3), 2))
        region = int(grid_y * 3 + grid_x)
        
        x_range = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
        y_range = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
        coverage = float((x_range * y_range) / (w * h))
        
        first_row_vec = corners_2d[self.chessboard_size[0]-1] - corners_2d[0]
        angle_rad = np.arctan2(first_row_vec[1], first_row_vec[0])
        angle_deg = abs(np.degrees(angle_rad))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        
        scale = float((x_range / w + y_range / h) / 2.0)
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'region': region,
            'coverage': coverage,
            'angle': angle_deg,
            'scale': scale
        }

    def analyze_images(self, input_dir):
        self.logger("=== 正在分析影像特徵 ===")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(Path(input_dir).glob(ext)))
            image_files.extend(list(Path(input_dir).glob(ext.upper())))
        image_files = sorted(list(set(image_files)))
        
        valid_data = []
        for idx, img_path in enumerate(image_files):
            img_path_str = str(img_path)
            # Handle Chinese paths
            img = cv2.imread(img_path_str)
            if img is None:
                try:
                    with open(img_path_str, 'rb') as f:
                        file_bytes = np.frombuffer(f.read(), np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                except Exception:
                    continue
            
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                feats = self.calculate_geometry_features(refined_corners, gray.shape)
                
                angle_bonus = 1.0 + (feats['angle'] / 30.0)
                score = (feats['coverage'] * 100) * angle_bonus
                
                valid_data.append({
                    'path': str(img_path),
                    'filename': img_path.name,
                    'corners': refined_corners,
                    'score': float(score),
                    'img_shape': gray.shape[::-1],
                    **feats
                })
                
                if idx % 50 == 0:
                    self.logger(f"已處理 {idx}/{len(image_files)} 張...")

        self.candidates = valid_data
        self.logger(f"分析完成：總計 {len(image_files)} 張，有效 {len(valid_data)} 張。")
        return len(valid_data)

    def calibrate_and_get_errors(self, selected_subset):
        if not selected_subset:
            return float('inf'), [], None, None

        objpoints = [self.objp for _ in selected_subset]
        imgpoints = [x['corners'] for x in selected_subset]
        img_shape = selected_subset[0]['img_shape']
        
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_shape, None, None
            )
            
            per_view_errors = []
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / np.sqrt(len(imgpoints2))
                per_view_errors.append(error)
                
            return ret, per_view_errors, mtx, dist
        except cv2.error:
            return float('inf'), [], None, None

    def select_best_images(self, target_rmse=0.5):
        if len(self.candidates) <= self.target_count:
            self.logger("候選圖片不足目標數量，回傳全部。")
            if len(self.candidates) > 0:
                rms, _, mtx, dist = self.calibrate_and_get_errors(self.candidates)
                return self.candidates, rms, mtx, dist
            return self.candidates, 0.0, None, None

        self.logger(f"\n=== 開始智能挑選 (目標: {self.target_count}張, RMS < {target_rmse}px) ===")
        
        sorted_candidates = sorted(self.candidates, key=lambda x: x['score'], reverse=True)
        selected = []
        region_map = defaultdict(list)
        for c in sorted_candidates:
            region_map[c['region']].append(c)
            
        for r in range(9):
            if region_map[r]:
                selected.append(region_map[r][0])
        
        needed = self.target_count - len(selected)
        if needed > 0:
            current_paths = {x['path'] for x in selected}
            pool = [x for x in sorted_candidates if x['path'] not in current_paths]
            
            if len(pool) > 0:
                feats = np.array([[x['angle'], x['scale']] for x in pool])
                feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-5)
                
                n_clusters = min(needed, len(pool))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
                
                cluster_groups = defaultdict(list)
                for idx, label in enumerate(kmeans.labels_):
                    cluster_groups[label].append(pool[idx])
                
                for lbl in range(n_clusters):
                    if cluster_groups[lbl]:
                        best_in_cluster = max(cluster_groups[lbl], key=lambda x: x['score'])
                        selected.append(best_in_cluster)
                        
                while len(selected) < self.target_count:
                    pool_rem = [x for x in sorted_candidates if x not in selected]
                    if not pool_rem:
                        break
                    selected.append(pool_rem[0])
        
        selected = selected[:self.target_count]
        
        current_rms, per_view_errors, mtx, dist = self.calibrate_and_get_errors(selected)
        self.logger(f"初始組合 RMS: {current_rms:.4f} px")
        
        iteration = 0
        max_iter = 50
        
        while current_rms > target_rmse and iteration < max_iter:
            iteration += 1
            
            worst_idx = np.argmax(per_view_errors)
            worst_img = selected[worst_idx]
            worst_error = per_view_errors[worst_idx]
            
            region_count = sum(1 for x in selected if x['region'] == worst_img['region'])
            is_single_region = (region_count == 1)
            
            pool = [x for x in self.candidates if x not in selected]
            same_region_pool = [x for x in pool if x['region'] == worst_img['region']]
            replacement = None
            
            if is_single_region:
                if same_region_pool:
                    replacement = max(same_region_pool, key=lambda x: x['score'])
                else:
                    # If this is the only image in the region and no replacement exists in the same region,
                    # try to find another image to remove that IS NOT the only one in its region.
                    skip_regions = {x['region'] for x in selected 
                                  if sum(1 for y in selected if y['region'] == x['region']) == 1}
                    available_indices = [i for i, x in enumerate(selected) 
                                       if x['region'] not in skip_regions]
                    if available_indices:
                        alt_worst_idx = max(available_indices, key=lambda i: per_view_errors[i])
                        alt_worst_img = selected[alt_worst_idx]
                        alt_same_region_pool = [x for x in pool if x['region'] == alt_worst_img['region']]
                        if alt_same_region_pool:
                            replacement = max(alt_same_region_pool, key=lambda x: x['score'])
                            worst_idx = alt_worst_idx
                            worst_img = alt_worst_img
                            worst_error = per_view_errors[alt_worst_idx]
                    if not replacement:
                        break
            else:
                if same_region_pool:
                    replacement = max(same_region_pool, key=lambda x: x['score'])
                elif pool:
                    replacement = max(pool, key=lambda x: x['score'])
                
            if replacement:
                new_selection = selected.copy()
                new_selection[worst_idx] = replacement
                
                regions_after = set(x['region'] for x in new_selection)
                if len(regions_after) < 9:
                    # Double check we didn't lose a region (should be covered by logic above, but safe to keep)
                    break
                
                new_rms, new_errors, n_mtx, n_dist = self.calibrate_and_get_errors(new_selection)
                
                if new_rms < current_rms:
                    self.logger(f"Iter {iteration}: 移除 {worst_img['filename']} (區域{worst_img['region']}, Err: {worst_error:.3f}) -> 替換為 {replacement['filename']} (區域{replacement['region']})。RMS 降至 {new_rms:.4f}")
                    selected = new_selection
                    current_rms = new_rms
                    per_view_errors = new_errors
                    mtx, dist = n_mtx, n_dist
                else:
                    break
            else:
                break
                
        final_regions = set(x['region'] for x in selected)
        if len(final_regions) < 9:
            self.logger(f"[警告] 最終選出的圖片只涵蓋 {len(final_regions)}/9 個區域")
        else:
            self.logger(f"[確認] 最終選出的圖片涵蓋全部 9 個區域")
            
        self.logger(f"最終 RMS: {current_rms:.4f} px")
        return selected, current_rms, mtx, dist

    def save_results(self, selected_images, final_rms, mtx, dist, output_dir):
        self.logger("\n=== 正在輸出挑選結果 ===")
        os.makedirs(output_dir, exist_ok=True)
        
        summary_data = []
        
        for i, img_data in enumerate(selected_images):
            try:
                new_name = f"calib_{i+1:02d}_{img_data['filename']}"
                dst_path = os.path.join(output_dir, new_name)
                shutil.copy2(img_data['path'], dst_path)
                
                summary_data.append({
                    'id': int(i+1),
                    'filename': str(new_name),
                    'original': str(img_data['filename']),
                    'region': int(img_data['region']),
                    'score': float(round(img_data['score'], 2)),
                })
            except Exception as e:
                self.logger(f"儲存圖片失敗 ({img_data['filename']}): {e}")

        # Note: We do NOT save npz here as the next step does the final calibration again according to user workflow
        # Only saving images to final_img folder
        
        try:
            self.plot_analysis(selected_images, final_rms, Path(output_dir).parent)
        except Exception as e:
            self.logger(f"繪圖失敗: {e}")

    def plot_analysis(self, images, rms, out_dir):
        regions = [x['region'] for x in images]
        angles = [x['angle'] for x in images]
        scores = [x['score'] for x in images]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        counts = np.zeros(9)
        for r in regions: counts[r] += 1
        axes[0].bar(range(9), counts, color='skyblue', edgecolor='black')
        axes[0].set_title('空間分佈 (3x3 Grid)')
        
        axes[1].hist(angles, bins=10, color='salmon', edgecolor='black', alpha=0.7)
        axes[1].set_title(f'角度分佈')
        
        axes[2].scatter(angles, scores, c='green', alpha=0.6)
        axes[2].set_title('評分 vs 角度')
        
        plt.suptitle(f'最終挑選分析 (RMS: {rms:.4f} px)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'single_selection_analysis.png'))
        plt.close(fig)

def run_auto_pick(input_dir, output_dir, chessboard_size=(9, 6), square_size=0.09, target_count=15, logger=print):
    """
    Step 4: Wrapper function to run the auto picker
    """
    selector = OptimizedSingleChessboardSelector(chessboard_size, square_size, target_count, logger)
    count = selector.analyze_images(input_dir)
    
    if count == 0:
        logger("未找到有效圖片。")
        return
        
    selected, final_rms, mtx, dist = selector.select_best_images(target_rmse=0.5)
    selector.save_results(selected, final_rms, mtx, dist, output_dir)
    return final_rms
