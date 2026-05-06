import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans

# 設置繪圖字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedStereoChessboardSelector:
    """
    智能雙目棋盤格影像挑選器
    透過空間分佈覆蓋、姿態多樣性抽樣與 RMS 疊代優化，挑選出最適合標定的立體影像對。
    """
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

    # ---------- 核心計算工具 ----------
    def calculate_static_reprojection_error(self, corners, mtx, dist):
        """利用 solvePnP 估算單張影像的重投影誤差"""
        if corners is None: return float('inf')
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners, mtx, dist)
        if not ret: return float('inf')
        p2, _ = cv2.projectPoints(self.objp, rvec, tvec, mtx, dist)
        return float(cv2.norm(corners, p2, cv2.NORM_L2) / len(p2))

    def calculate_geometry_features(self, corners, image_shape):
        """計算影像幾何特徵：區域(3x3)、覆蓋率、傾斜角度與尺度"""
        pts = corners.reshape(-1, 2)
        h, w = image_shape[:2]
        cx, cy = np.mean(pts, axis=0) / [w, h]
        region = int(min(int(cy * 3), 2) * 3 + min(int(cx * 3), 2))
        
        x_rng, y_rng = np.max(pts, axis=0) - np.min(pts, axis=0)
        vec = pts[self.chessboard_size[0]-1] - pts[0]
        angle = abs(np.degrees(np.arctan2(vec[1], vec[0])))
        if angle > 90: angle = 180 - angle
        
        return {
            'center_x': float(cx), 'center_y': float(cy), 'region': region,
            'coverage': float((x_rng * y_rng) / (w * h)), 'angle': float(angle),
            'scale': float((x_rng / w + y_rng / h) / 2.0)
        }

    def _detect_single(self, img_path: Path, is_left: bool):
        """單張影像檢測：提取角點並計算評分"""
        img = None
        try:
            with open(str(img_path), 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception: pass
        
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if not ret: return None
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        feats = self.calculate_geometry_features(refined, gray.shape)
        
        mtx, dist = (self.mtxL, self.distL) if is_left else (self.mtxR, self.distR)
        static_err = self.calculate_static_reprojection_error(refined, mtx, dist)
        
        score = (feats['coverage'] * 100) * (1.0 + feats['angle'] / 90.0) * (1.0 / max(static_err, 1.0))
        
        return {
            'path': str(img_path), 'filename': img_path.name, 'corners': refined,
            'score': float(score), 'static_error': float(static_err), **feats
        }

    def analyze_and_pair(self, left_dir: str, right_dir: str, logger=print):
        """同步配對左右影像並執行初步檢測"""
        logger("正在分析影像特徵與配對...")
        l_files = sorted(list(Path(left_dir).glob('*.[jJpP]*')))
        r_map = {p.name: p for p in Path(right_dir).glob('*.[jJpP]*')}
        
        paired = []
        for lf in l_files:
            rf = r_map.get(lf.name)
            if not rf: continue
            
            li, ri = self._detect_single(lf, True), self._detect_single(rf, False)
            if li and ri and li['static_error'] < 10.0 and ri['static_error'] < 10.0:
                paired.append({
                    'pair_key': lf.name, 'left': li, 'right': ri,
                    'pair_score': float(min(li['score'], ri['score'])),
                    'pair_region': int(li['region']),
                    'static_error_sum': float(li['static_error'] + ri['static_error'])
                })
        
        self.paired_infos = paired
        logger(f"有效檢測配對數: {len(paired)}")
        return len(paired)

    def perform_stereo_calibration(self, selected_pairs):
        """執行雙目相機初步標定並返回 RMS 與參數"""
        if not selected_pairs: return float('inf'), None, None, None, None, None, None
        
        objpoints = [self.objp] * len(selected_pairs)
        imgptsL = [p['left']['corners'] for p in selected_pairs]
        imgptsR = [p['right']['corners'] for p in selected_pairs]
        
        try:
            with open(selected_pairs[0]['left']['path'], 'rb') as f:
                img_size = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE).shape[::-1]
            
            ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgptsL, imgptsR, self.mtxL, self.distL, self.mtxR, self.distR,
                img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
            return float(ret), M1, D1, M2, D2, R, T
        except Exception:
            return float('inf'), None, None, None, None, None, None

    def select_best_pairs(self, target_rmse=0.5, logger=print):
        """雙目智能挑選演算法：區域填充 -> K-Means 聚類 -> RMS 疊代優化"""
        if len(self.paired_infos) < self.target_pairs:
            logger("可用配對不足，回傳全部。")
            return self.paired_infos, 0.0

        logger(f"\n=== 開始智能挑選 (目標: {self.target_pairs}對) ===")
        candidates = sorted(self.paired_infos, key=lambda x: x['pair_score'], reverse=True)
        
        # 1. 空間分佈優先
        selected, r_map = [], defaultdict(list)
        for p in candidates: r_map[p['pair_region']].append(p)
        for r in range(9):
            if r_map[r]: selected.append(r_map[r][0])
        
        # 2. 多樣性填充 (K-Means)
        needed = self.target_pairs - len(selected)
        if needed > 0:
            pool = [x for x in candidates if x not in selected]
            if pool:
                feats = np.array([[x['left']['angle'], x['right']['angle'], x['left']['scale']] for x in pool])
                feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-5)
                n_clusters = min(needed, len(pool))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
                
                c_groups = defaultdict(list)
                for i, l in enumerate(kmeans.labels_): c_groups[l].append(pool[i])
                for l in range(n_clusters):
                    if c_groups[l]: selected.append(max(c_groups[l], key=lambda x: x['pair_score']))
            
            while len(selected) < self.target_pairs:
                rem = [x for x in candidates if x not in selected]
                if not rem: break
                selected.append(rem[0])
            
        selected = selected[:self.target_pairs]
        rms, M1, D1, M2, D2, R, T = self.perform_stereo_calibration(selected)
        logger(f"初始組合 RMS: {rms:.4f} px")
        
        # 3. 疊代優化 (區域鎖定)
        for it in range(50):
            if rms <= target_rmse: break
            
            # 評估每對影像的誤差貢獻
            errs = []
            for p in selected:
                rL, rvL, tvL = cv2.solvePnP(self.objp, p['left']['corners'], M1, D1)
                rR, rvR, tvR = cv2.solvePnP(self.objp, p['right']['corners'], M2, D2)
                if rL and rR:
                    pL, _ = cv2.projectPoints(self.objp, rvL, tvL, M1, D1)
                    pR, _ = cv2.projectPoints(self.objp, rvR, tvR, M2, D2)
                    errs.append((cv2.norm(p['left']['corners'], pL, cv2.NORM_L2) + cv2.norm(p['right']['corners'], pR, cv2.NORM_L2)) / (2.0 * np.sqrt(len(pL))))
                else: errs.append(float('inf'))
            
            worst_idx = np.argmax(errs)
            worst_p = selected[worst_idx]
            pool = [x for x in candidates if x not in selected]
            
            # 優先找同區域替換以維持空間分佈
            repl_pool = [x for x in pool if x['pair_region'] == worst_p['pair_region']] or pool
            if not repl_pool: break
            
            repl = max(repl_pool, key=lambda x: x['pair_score'])
            new_sel = selected.copy(); new_sel[worst_idx] = repl
            n_rms, nM1, nD1, nM2, nD2, nR, nT = self.perform_stereo_calibration(new_sel)
            
            if n_rms < rms:
                logger(f"Iter {it+1}: 替換 {worst_p['pair_key']} -> {repl['pair_key']}，RMS: {n_rms:.4f}")
                selected, rms, M1, D1, M2, D2, R, T = new_sel, n_rms, nM1, nD1, nM2, nD2, nR, nT
            else: break
                
        logger(f"最終 RMS: {rms:.4f} px")
        
        final_regions = set(p['pair_region'] for p in selected)
        missing = set(range(9)) - final_regions
        if missing: logger(f"警告: 缺少區域: {missing}")
        else: logger("確認: 覆蓋所有 9 個區域。")
            
        return selected, rms

    def save_results(self, selected_pairs, final_rms, out_left, out_right, logger=print):
        """將挑選後的影像配對複製到目標資料夾並產出 JSON 報告"""
        self.logger = logger
        self.logger("\n=== 開始輸出結果 ===")
        os.makedirs(out_left, exist_ok=True); os.makedirs(out_right, exist_ok=True)
        
        summary = []
        for i, p in enumerate(selected_pairs):
            try:
                fn_l = f"stereo_{i+1:02d}_L_{p['left']['filename']}"
                fn_r = f"stereo_{i+1:02d}_R_{p['right']['filename']}"
                
                for src, dst in [(p['left']['path'], os.path.join(out_left, fn_l)), (p['right']['path'], os.path.join(out_right, fn_r))]:
                    with open(src, 'rb') as f:
                        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        _, enc = cv2.imencode('.jpg', img)
                        with open(dst, 'wb') as f: f.write(enc.tobytes())
                
                summary.append({
                    'id': i+1, 'file': p['pair_key'], 'region': p['pair_region'],
                    'score': round(p['pair_score'], 2), 'err': round(p['static_error_sum'], 3)
                })
            except Exception as e: self.logger(f"儲存失敗 (ID {i+1}): {e}")

        # 輸出 JSON 報表與分析圖
        report_path = os.path.join(Path(out_left).parent.parent, 'calibration_selection_report.json')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({'total': len(selected_pairs), 'rms': round(final_rms, 4), 'details': summary}, f, indent=2, ensure_ascii=False)
        except Exception: pass
        
        try: self.plot_analysis(selected_pairs, final_rms, Path(out_left).parent.parent)
        except Exception: pass

    def plot_analysis(self, pairs, rms, out_dir):
        """繪製分析圖表：空間分佈與角度分佈"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts = np.zeros(9)
        for r in [p['pair_region'] for p in pairs]: counts[r] += 1
        axes[0].bar(range(9), counts, color='skyblue', edgecolor='black')
        axes[0].set_title('空間分佈 (3x3 Grid)'); axes[0].set_xticks(range(9))
        
        axes[1].hist([p['left']['angle'] for p in pairs], bins=10, color='salmon', alpha=0.7)
        axes[1].set_title(f'傾斜角度分佈 (RMS: {rms:.3f}px)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'selection_analysis.png')); plt.close(fig)

def run_stereo_pick(left_dir, right_dir, out_left, out_right, mtxL, distL, mtxR, distR, chessboard_size=(9, 6), target_pairs=15, logger=print):
    """雙目影像挑選主進入點"""
    sel = OptimizedStereoChessboardSelector(mtxL, distL, mtxR, distR, chessboard_size, target_pairs)
    if sel.analyze_and_pair(left_dir, right_dir, logger) == 0: return 0, 0.0
    selected, rms = sel.select_best_pairs(target_rmse=0.5, logger=logger)
    sel.save_results(selected, rms, out_left, out_right, logger)
    return len(selected), rms