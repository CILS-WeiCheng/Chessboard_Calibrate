import cv2
import math
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
    核心邏輯：結合空間分佈 (3x3 Grid)、姿態多樣性 (K-Means)
    與 Jacobian 條件數 (Condition Number) 疊代優化。
    """

    def __init__(self, chessboard_size=(9, 6), square_size=0.09,
                 target_count=15, logger=print):
        self.chessboard_size = chessboard_size
        self.target_count = target_count
        self.square_size = square_size
        self.logger = logger
        self.candidates = []

        self.objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3),
            np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0:self.chessboard_size[0],
            0:self.chessboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= self.square_size

    # ─────────────────────────────────────────────────────────
    # 特徵計算
    # ─────────────────────────────────────────────────────────
    def calculate_geometry_features(self, corners, image_shape):
        """計算棋盤格幾何特徵：區域、面積覆蓋、透視變形量與縮放比例"""
        corners_2d = corners.reshape(-1, 2)
        h, w = image_shape[:2]

        # 區域計算 (3x3 Grid)
        center_x, center_y = np.mean(corners_2d, axis=0) / [w, h]
        grid_x = int(min(center_x * 3, 2))
        grid_y = int(min(center_y * 3, 2))
        region = int(grid_y * 3 + grid_x)

        # 幾何特性
        x_range = np.ptp(corners_2d[:, 0])
        y_range = np.ptp(corners_2d[:, 1])
        coverage = float((x_range * y_range) / (w * h))

        W, H = self.chessboard_size
        top_left = corners_2d[0]
        top_right = corners_2d[W - 1]
        bottom_left = corners_2d[(H - 1) * W]
        bottom_right = corners_2d[H * W - 1]

        # 透視變形量 (tilt)：衡量 out-of-plane 傾斜，對焦距計算幫助極大
        top_len = np.linalg.norm(top_right - top_left)
        bottom_len = np.linalg.norm(bottom_right - bottom_left)
        left_len = np.linalg.norm(bottom_left - top_left)
        right_len = np.linalg.norm(bottom_right - top_right)

        tilt_x = abs(top_len - bottom_len) / max(top_len, bottom_len, 1e-5)
        tilt_y = abs(left_len - right_len) / max(left_len, right_len, 1e-5)
        tilt = float(tilt_x + tilt_y)

        # 保留平面旋轉角以供 K-Means 增加多樣性
        vec = top_right - top_left
        angle = np.degrees(np.arctan2(vec[1], vec[0]))

        return {
            'center_x': center_x, 'center_y': center_y, 'region': region,
            'coverage': coverage, 'tilt': tilt, 'angle': float(angle),
            'scale': float((x_range / w + y_range / h) / 2)
        }

    # ─────────────────────────────────────────────────────────
    # 影像分析
    # ─────────────────────────────────────────────────────────
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
                        img = cv2.imdecode(
                            np.frombuffer(f.read(), np.uint8),
                            cv2.IMREAD_COLOR
                        )
                except Exception:
                    continue

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, self.chessboard_size, None
            )

            if ret:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001
                )
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                feats = self.calculate_geometry_features(refined, gray.shape)

                # 評分機制：
                # 1. 面積覆蓋率使用 log-scale，避免過大面積主導分數
                # 2. 透視變形獎勵：tilt 越高代表有出平面傾斜，
                #    對焦距計算幫助極大
                # 3. 變形不足時施加懲罰，避免只選到純正面（平行）圖片
                coverage_score = math.log1p(feats['coverage'] * 100)
                tilt = feats['tilt']
                tilt_bonus = min(tilt / 0.15, 1.0)
                tilt_penalty = max(0.0, 1.0 - tilt / 0.03) * 0.5
                score = coverage_score * (1.0 + tilt_bonus) * (1.0 - tilt_penalty)

                valid_data.append({
                    'path': str(img_path), 'filename': img_path.name,
                    'corners': refined, 'score': float(score),
                    'img_shape': gray.shape[::-1], **feats
                })

                if idx % 50 == 0:
                    self.logger(f"已處理 {idx}/{len(image_files)} 張...")

        self.candidates = valid_data
        self.logger(
            f"分析完成：共計 {len(image_files)} 張，有效 {len(valid_data)} 張。"
        )
        return len(valid_data)

    # ─────────────────────────────────────────────────────────
    # 標定與評估
    # ─────────────────────────────────────────────────────────
    def perform_calibration(self, selected_data):
        """執行核心標定計算"""
        objpoints = [self.objp] * len(selected_data)
        imgpoints = [d['corners'] for d in selected_data]
        img_size = selected_data[0]['img_shape']

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001
        )
        # 鎖定 K3 並將切向畸變歸零，以防止過擬合與降低條件數
        flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

        return cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None,
            criteria=criteria, flags=flags
        )

    def calibrate_and_get_errors(self, selected_subset):
        """計算給定組合的 RMS 與單張投影誤差"""
        if not selected_subset:
            return float('inf'), [], None, None
        try:
            ret, mtx, dist, rvecs, tvecs = self.perform_calibration(
                selected_subset
            )
            per_view_errors = []
            for i in range(len(selected_subset)):
                p2, _ = cv2.projectPoints(
                    self.objp, rvecs[i], tvecs[i], mtx, dist
                )
                err = cv2.norm(
                    selected_subset[i]['corners'], p2, cv2.NORM_L2
                ) / np.sqrt(len(p2))
                per_view_errors.append(err)
            return ret, per_view_errors, mtx, dist
        except cv2.error:
            return float('inf'), [], None, None

    def evaluate_condition_number(self, selected_subset):
        """
        計算標定雅可比矩陣的條件數 (Condition Number) 與凸包覆蓋率 (Convex Hull Coverage)。
        條件數代表內參的穩定性/可觀測性。
        """
        if len(selected_subset) < 3:
            return float('inf'), float('inf'), 0.0, None, None
        try:
            ret, mtx, dist, rvecs, tvecs = self.perform_calibration(
                selected_subset
            )

            # 計算空間角點的凸包覆蓋率
            all_pts = np.vstack([d['corners'].reshape(-1, 2) for d in selected_subset]).astype(np.float32)
            hull = cv2.convexHull(all_pts)
            hull_area = cv2.contourArea(hull)
            img_w, img_h = selected_subset[0]['img_shape']
            coverage_ratio = float(hull_area / (img_w * img_h))

            # 建立堆疊的雅可比矩陣
            J_int_list = []
            for i in range(len(selected_subset)):
                # cv2.projectPoints 輸出的 Jacobian shape = (2N, 14)
                # 欄位排列: [drvec(3), dtvec(3), dfx, dfy, dcx, dcy,
                #            dk1, dk2, dp1, dp2]
                # 第 6~13 欄是內參 (fx, fy, cx, cy, k1, k2, p1, p2)
                _, jac = cv2.projectPoints(
                    self.objp, rvecs[i], tvecs[i], mtx, dist
                )
                J_int = jac[:, 6:14]
                J_int_list.append(J_int)

            J_stack = np.vstack(J_int_list)

            # Column Normalization：
            # 消除焦距 (千位數) 與畸變 (小數) 的物理單位差異
            norms = np.linalg.norm(J_stack, axis=0)
            norms[norms == 0] = 1.0  # 避免除以零 (若某個參數被鎖死)
            J_norm = J_stack / norms

            # 進行奇異值分解 (SVD)
            _, S, _ = np.linalg.svd(J_norm, full_matrices=False)
            S_valid = S[S > 1e-5]  # 過濾掉無效(被鎖死)的維度

            if len(S_valid) > 0:
                cond_num = float(S_valid[0] / S_valid[-1])
            else:
                cond_num = float('inf')

            return cond_num, ret, coverage_ratio, mtx, dist

        except cv2.error:
            return float('inf'), float('inf'), 0.0, None, None

    # ─────────────────────────────────────────────────────────
    # 核心挑選演算法
    # ─────────────────────────────────────────────────────────
    def select_best_images(self, target_rmse=0.5):
        """
        核心挑選演算法：
        1. 區域填充 → 確保空間分佈
        2. K-Means 多樣性抽樣 → 確保姿態多樣
        3. 條件數 (Condition Number) 疊代優化 → 最大化幾何約束力，並受限於角點包絡覆蓋率
        """
        if len(self.candidates) <= self.target_count:
            if not self.candidates:
                return [], 0.0, float('inf'), None, None
            cond, rms, cov, mtx, dist = self.evaluate_condition_number(
                self.candidates
            )
            return self.candidates, rms, cond, mtx, dist

        self.logger(
            f"\n=== 開始智能挑選 (目標: {self.target_count}張, "
            f"最小化條件數, RMS安全上限: {target_rmse}px) ==="
        )

        # 1. 空間分佈優先：確保 9 個區域都有代表影像
        sorted_candidates = sorted(
            self.candidates, key=lambda x: x['score'], reverse=True
        )
        selected = []
        region_map = defaultdict(list)
        for c in sorted_candidates:
            region_map[c['region']].append(c)
        for r in range(9):
            if region_map[r]:
                selected.append(region_map[r][0])

        # 2. 補充多樣性：利用 K-Means 在剩餘影像中按姿態聚類抽樣
        needed = self.target_count - len(selected)
        if needed > 0:
            current_paths = {x['path'] for x in selected}
            pool = [
                x for x in sorted_candidates
                if x['path'] not in current_paths
            ]
            if pool:
                feats = np.array([
                    [x['tilt'], x['angle'], x['scale'],
                     x['center_x'], x['center_y']]
                    for x in pool
                ])
                feats = (feats - feats.mean(axis=0)) / (
                    feats.std(axis=0) + 1e-5
                )
                n_clusters = min(needed, len(pool))
                kmeans = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10
                ).fit(feats)

                cluster_groups = defaultdict(list)
                for idx, label in enumerate(kmeans.labels_):
                    cluster_groups[label].append(pool[idx])
                for lbl in range(n_clusters):
                    if cluster_groups[lbl]:
                        selected.append(
                            max(cluster_groups[lbl],
                                key=lambda x: x['score'])
                        )

            while len(selected) < self.target_count:
                rem = [x for x in sorted_candidates if x not in selected]
                if not rem:
                    break
                selected.append(rem[0])

        selected = selected[:self.target_count]
        current_cond, current_rms, current_cov, mtx, dist = (
            self.evaluate_condition_number(selected)
        )
        self.logger(
            f"初始組合 | 條件數: {current_cond:.2f} | "
            f"RMS: {current_rms:.4f} px | 凸包覆蓋率: {current_cov*100:.1f}%"
        )

        # 3. 疊代優化：以「最小化條件數」為目標進行隨機退火探索
        max_iter = 100
        for iter_count in range(1, max_iter + 1):
            # 隨機選一張淘汰
            swap_out_idx = np.random.randint(len(selected))

            # 從未選中的候選池中隨機選一張替補
            pool = [x for x in self.candidates if x not in selected]
            if not pool:
                break

            # 使用帶有溫度參數 (T=0.2) 的 Softmax，增加對高分影像的選擇概率
            temp = 0.2
            scores = np.array([x['score'] for x in pool])
            scores = (scores - scores.max()) / temp  # 數值穩定：防止 exp 溢位
            weights = np.exp(scores)
            weights = weights / weights.sum()

            # 先選 index 再取值，避免 np.random.choice 對 dict list 的不穩定行為
            chosen_idx = np.random.choice(len(pool), p=weights)
            img_in = pool[chosen_idx]

            new_selection = selected.copy()
            new_selection[swap_out_idx] = img_in

            new_cond, new_rms, new_cov, n_mtx, n_dist = (
                self.evaluate_condition_number(new_selection)
            )

            # 接受條件：
            # 1. 條件數下降
            # 2. RMS 依然在安全範圍內
            # 3. 空間凸包覆蓋率不得出現嚴重退化 (不低於 65%，且不比當前差超過 2%)
            cov_ok = (new_cov >= 0.65) or (new_cov >= current_cov * 0.98)
            if new_cond < current_cond and new_rms < target_rmse and cov_ok:
                self.logger(
                    f"Iter {iter_count:3d}: 替換成功 | "
                    f"Cond: {current_cond:.2f} -> {new_cond:.2f} | "
                    f"RMS: {new_rms:.4f} px | 覆蓋率: {new_cov*100:.1f}%"
                )
                selected = new_selection
                current_cond = new_cond
                current_rms = new_rms
                current_cov = new_cov
                mtx = n_mtx
                dist = n_dist

        self.logger(
            f"最終組合 | 條件數: {current_cond:.2f} | "
            f"RMS: {current_rms:.4f} px | 凸包覆蓋率: {current_cov*100:.1f}%"
        )
        return selected, current_rms, current_cond, mtx, dist

    # ─────────────────────────────────────────────────────────
    # 結果輸出
    # ─────────────────────────────────────────────────────────
    def save_results(self, selected_images, final_rms, final_cond,
                     mtx, dist, output_dir):
        """將挑選後的影像複製到目標資料夾並繪製分析圖"""
        self.logger("\n=== 正在輸出挑選結果 ===")
        os.makedirs(output_dir, exist_ok=True)
        for i, img_data in enumerate(selected_images):
            try:
                new_name = f"calib_{i+1:02d}_{img_data['filename']}"
                shutil.copy2(
                    img_data['path'], os.path.join(output_dir, new_name)
                )
            except Exception as e:
                self.logger(f"儲存失敗 ({img_data['filename']}): {e}")

        try:
            self.plot_analysis(
                selected_images, final_rms, final_cond,
                Path(output_dir).parent
            )
        except Exception:
            pass

    def plot_analysis(self, images, rms, cond, out_dir):
        """繪製分佈圖：區域、角度、評分"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 計算最終的凸包覆蓋率
        all_pts = np.vstack([x['corners'].reshape(-1, 2) for x in images]).astype(np.float32)
        hull = cv2.convexHull(all_pts)
        hull_area = cv2.contourArea(hull)
        img_w, img_h = images[0]['img_shape']
        cov_ratio = float(hull_area / (img_w * img_h))

        counts = np.zeros(9)
        for r in [x['region'] for x in images]:
            counts[r] += 1
        axes[0].bar(range(9), counts, color='skyblue', edgecolor='black')
        axes[0].set_title('空間分佈 (3x3 Grid)')

        axes[1].hist(
            [x['angle'] for x in images], bins=10,
            color='salmon', alpha=0.7
        )
        axes[1].set_title('角度分佈')

        axes[2].scatter(
            [x['angle'] for x in images],
            [x['score'] for x in images],
            c='green', alpha=0.6
        )
        axes[2].set_title('評分 vs 角度')

        plt.suptitle(
            f'最終挑選分析 (Cond: {cond:.2f} | RMS: {rms:.4f} px | 覆蓋率: {cov_ratio*100:.1f}%)',
            fontsize=14
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'single_selection_analysis.png'))
        plt.close(fig)


def run_auto_pick(input_dir, output_dir, chessboard_size=(9, 6),
                  square_size=0.09, target_count=15, logger=print):
    """
    Wrapper function：執行完整的自動挑選流程。
    回傳 (final_rms, final_cond) 元組。
    """
    selector = OptimizedSingleChessboardSelector(
        chessboard_size, square_size, target_count, logger
    )
    count = selector.analyze_images(input_dir)

    if count == 0:
        logger("未找到有效圖片。")
        return None, None

    selected, final_rms, final_cond, mtx, dist = (
        selector.select_best_images(target_rmse=0.5)
    )
    selector.save_results(
        selected, final_rms, final_cond, mtx, dist, output_dir
    )
    return final_rms, final_cond
