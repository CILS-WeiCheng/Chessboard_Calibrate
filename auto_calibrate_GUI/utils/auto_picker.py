import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans

# 使用 Agg 後端以確保在多執行緒中穩定繪圖
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

class OptimizedSingleChessboardSelector:
    """
    單目相機棋盤格智能挑選器
    核心邏輯：結合空間分佈 (3x3 Grid)、姿態多樣性 (K-Means) 與 RMS 疊代優化。
    """
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
        """計算棋盤格幾何特徵：區域、面積覆蓋、傾斜角度與縮放比例"""
        corners_2d = corners.reshape(-1, 2)
        h, w = image_shape[:2]
        
        # 區域計算 (3x3 Grid)
        center_x, center_y = np.mean(corners_2d, axis=0) / [w, h]
        grid_x, grid_y = int(min(center_x * 3, 2)), int(min(center_y * 3, 2))
        region = int(grid_y * 3 + grid_x)
        
        # 幾何特性
        x_range = np.ptp(corners_2d[:, 0])
        y_range = np.ptp(corners_2d[:, 1])
        coverage = float((x_range * y_range) / (w * h))
        
        vec = corners_2d[self.chessboard_size[0]-1] - corners_2d[0]
        angle = abs(np.degrees(np.arctan2(vec[1], vec[0])))
        if angle > 90: angle = 180 - angle
        
        return {'center_x': center_x, 'center_y': center_y, 'region': region,
                'coverage': coverage, 'angle': angle, 'scale': float((x_range/w + y_range/h)/2)}

    def analyze_images(self, input_dir):
        """批次讀取並分析影像，提取候選棋盤格數據"""
        self.logger("=== 正在分析影像特徵 ===")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(Path(input_dir).glob(ext)))
            image_files.extend(list(Path(input_dir).glob(ext.upper())))
        image_files = sorted(list(set(image_files)))
        
        valid_data = []
        for idx, img_path in enumerate(image_files):
            img_path_str = str(img_path)
            # 支援中文字元路徑
            img = cv2.imread(img_path_str)
            if img is None:
                try:
                    with open(img_path_str, 'rb') as f:
                        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                except Exception: continue
            
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                feats = self.calculate_geometry_features(refined, gray.shape)
                
                score = (feats['coverage'] * 100) * (1.0 + feats['angle'] / 30.0)
                
                valid_data.append({
                    'path': str(img_path), 'filename': img_path.name,
                    'corners': refined, 'score': float(score),
                    'img_shape': gray.shape[::-1], **feats
                })
                
                if idx % 50 == 0: self.logger(f"已處理 {idx}/{len(image_files)} 張...")

        self.candidates = valid_data
        self.logger(f"分析完成：共計 {len(image_files)} 張，有效 {len(valid_data)} 張。")
        return len(valid_data)

    def perform_calibration(self, selected_data):
        """執行核心標定計算"""
        objpoints = [self.objp] * len(selected_data)
        imgpoints = [d['corners'] for d in selected_data]
        img_size = selected_data[0]['img_shape']
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        flags = cv2.CALIB_FIX_K3
        
        return cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None,
                                   criteria=criteria, flags=flags)

    def calibrate_and_get_errors(self, selected_subset):
        """計算給定組合的 RMS 與單張投影誤差"""
        if not selected_subset: return float('inf'), [], None, None
        try:
            ret, mtx, dist, rvecs, tvecs = self.perform_calibration(selected_subset)
            per_view_errors = []
            for i in range(len(selected_subset)):
                p2, _ = cv2.projectPoints(self.objp, rvecs[i], tvecs[i], mtx, dist)
                err = cv2.norm(selected_subset[i]['corners'], p2, cv2.NORM_L2) / np.sqrt(len(p2))
                per_view_errors.append(err)
            return ret, per_view_errors, mtx, dist
        except cv2.error:
            return float('inf'), [], None, None

    def select_best_images(self, target_rmse=0.1):
        """核心挑選演算法：區域填充 -> K-Means 多樣性抽樣 -> RMS 疊代優化"""
        if len(self.candidates) <= self.target_count:
            if not self.candidates: return [], 0.0, None, None
            rms, _, mtx, dist = self.calibrate_and_get_errors(self.candidates)
            return self.candidates, rms, mtx, dist

        self.logger(f"\n=== 開始智能挑選 (目標: {self.target_count}張, RMS < {target_rmse}px) ===")
        
        # 1. 空間分佈優先：確保 9 個區域都有代表影像
        sorted_candidates = sorted(self.candidates, key=lambda x: x['score'], reverse=True)
        selected = []
        region_map = defaultdict(list)
        for c in sorted_candidates: region_map[c['region']].append(c)
        for r in range(9):
            if region_map[r]: selected.append(region_map[r][0])
        
        # 2. 補充多樣性：利用 K-Means 在剩餘影像中按姿態聚類抽樣
        needed = self.target_count - len(selected)
        if needed > 0:
            current_paths = {x['path'] for x in selected}
            pool = [x for x in sorted_candidates if x['path'] not in current_paths]
            if pool:
                feats = np.array([[x['angle'], x['scale']] for x in pool])
                feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-5)
                n_clusters = min(needed, len(pool))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
                
                cluster_groups = defaultdict(list)
                for idx, label in enumerate(kmeans.labels_): cluster_groups[label].append(pool[idx])
                for lbl in range(n_clusters):
                    if cluster_groups[lbl]:
                        selected.append(max(cluster_groups[lbl], key=lambda x: x['score']))
            
            while len(selected) < self.target_count:
                rem = [x for x in sorted_candidates if x not in selected]
                if not rem: break
                selected.append(rem[0])
        
        selected = selected[:self.target_count]
        current_rms, per_view_errors, mtx, dist = self.calibrate_and_get_errors(selected)
        self.logger(f"初始組合 RMS: {current_rms:.4f} px")
        
        # 3. 疊代優化：循環剔除誤差最大的影像並嘗試替換
        iter_count, max_iter = 0, 50
        while current_rms > target_rmse and iter_count < max_iter:
            iter_count += 1
            worst_idx = np.argmax(per_view_errors)
            worst_img = selected[worst_idx]
            
            pool = [x for x in self.candidates if x not in selected]
            same_region_pool = [x for x in pool if x['region'] == worst_img['region']]
            
            replacement = None
            if same_region_pool:
                replacement = max(same_region_pool, key=lambda x: x['score'])
            elif pool:
                replacement = max(pool, key=lambda x: x['score'])
                
            if replacement:
                new_selection = selected.copy()
                new_selection[worst_idx] = replacement
                new_rms, new_errors, n_mtx, n_dist = self.calibrate_and_get_errors(new_selection)
                
                if new_rms < current_rms:
                    self.logger(f"Iter {iter_count}: 替換 {worst_img['filename']} -> {replacement['filename']}，RMS: {new_rms:.4f}")
                    selected, current_rms, per_view_errors, mtx, dist = new_selection, new_rms, new_errors, n_mtx, n_dist
                else: break
            else: break
            
        self.logger(f"最終 RMS: {current_rms:.4f} px")
        return selected, current_rms, mtx, dist

    def save_results(self, selected_images, final_rms, mtx, dist, output_dir):
        """將挑選後的影像複製到目標資料夾並繪製分析圖"""
        self.logger("\n=== 正在輸出挑選結果 ===")
        os.makedirs(output_dir, exist_ok=True)
        for i, img_data in enumerate(selected_images):
            try:
                new_name = f"calib_{i+1:02d}_{img_data['filename']}"
                shutil.copy2(img_data['path'], os.path.join(output_dir, new_name))
            except Exception as e:
                self.logger(f"儲存失敗 ({img_data['filename']}): {e}")
        
        try:
            self.plot_analysis(selected_images, final_rms, Path(output_dir).parent)
        except Exception: pass

    def plot_analysis(self, images, rms, out_dir):
        """繪製分佈圖：區域、角度、評分"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        counts = np.zeros(9)
        for r in [x['region'] for x in images]: counts[r] += 1
        axes[0].bar(range(9), counts, color='skyblue', edgecolor='black')
        axes[0].set_title('空間分佈 (3x3 Grid)')
        
        axes[1].hist([x['angle'] for x in images], bins=10, color='salmon', alpha=0.7)
        axes[1].set_title('角度分佈')
        
        axes[2].scatter([x['angle'] for x in images], [x['score'] for x in images], c='green', alpha=0.6)
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
        
    selected, final_rms, mtx, dist = selector.select_best_images(target_rmse=0.1)
    selector.save_results(selected, final_rms, mtx, dist, output_dir)
    return final_rms
