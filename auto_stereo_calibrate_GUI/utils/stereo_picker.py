"""
stereo_picker.py
================
智能雙目棋盤格影像挑選器。

優化歷程：
    一、效能優化  - 多執行緒檢測、image_size 快取、容忍探索疊代
    二、演算法品質 - log-scale 評分、豐富 KMeans 特徵、Simulated Annealing 替換
    三、覆蓋多樣性 - 5×5 細化網格、有符號角度
    四、GUI 穩健性 - (由 gui.py 負責)
    五、可維護性   - square_size 參數化、統一 logger、完整型別提示
"""

import json
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# ── 繪圖字型設定 ──────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ── 常數 ─────────────────────────────────────────────────────────────────────
# 5×5 網格共 25 個空間區域（優化 8）
_GRID_N: int = 5
_GRID_CELLS: int = _GRID_N * _GRID_N  # 25

# Simulated Annealing 初始溫度與冷卻率（優化 6）
_SA_INIT_TEMP: float = 0.5
_SA_COOLING: float = 0.85

# 連續無改善次數上限（優化 3）
_MAX_NO_IMPROVE: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
class OptimizedStereoChessboardSelector:
    """
    智能雙目棋盤格影像挑選器。

    透過空間分佈覆蓋（5×5 網格）、姿態多樣性抽樣（KMeans）
    與帶 Simulated Annealing 的 RMS 疊代優化，挑選最適合標定的立體影像對。

    Parameters
    ----------
    mtxL, distL : np.ndarray
        左相機內參與畸變係數。
    mtxR, distR : np.ndarray
        右相機內參與畸變係數。
    chessboard_size : Tuple[int, int]
        棋盤格角點數 (cols, rows)，預設 (9, 6)。
    target_pairs : int
        目標挑選影像對數，預設 15。
    square_size : float
        棋盤格方格邊長（公尺），預設 0.09。
    """

    def __init__(
        self,
        mtxL: np.ndarray,
        distL: np.ndarray,
        mtxR: np.ndarray,
        distR: np.ndarray,
        chessboard_size: Tuple[int, int] = (9, 6),
        target_pairs: int = 15,
        square_size: float = 0.09,   # 優化 14：由外部傳入，不再硬編碼
    ) -> None:
        self.chessboard_size = chessboard_size
        self.target_pairs = target_pairs
        self.square_size = square_size
        self.paired_infos: List[Dict[str, Any]] = []

        # 相機參數（作為初始估計基準）
        self.mtxL = mtxL
        self.distL = distL
        self.mtxR = mtxR
        self.distR = distR

        # 建立物件點（世界座標）
        self.objp: np.ndarray = np.zeros(
            (chessboard_size[0] * chessboard_size[1], 3), np.float32
        )
        self.objp[:, :2] = (
            np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        )
        self.objp *= square_size

    # ── 核心計算工具 ───────────────────────────────────────────────────────────

    def calculate_static_reprojection_error(
        self,
        corners: np.ndarray,
        mtx: np.ndarray,
        dist: np.ndarray,
    ) -> float:
        """利用 solvePnP 估算單張影像的重投影誤差（單位：像素）。"""
        if corners is None:
            return float('inf')
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners, mtx, dist)
        if not ret:
            return float('inf')
        p2, _ = cv2.projectPoints(self.objp, rvec, tvec, mtx, dist)
        return float(cv2.norm(corners, p2, cv2.NORM_L2) / len(p2))

    def calculate_geometry_features(
        self,
        corners: np.ndarray,
        image_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """
        計算影像幾何特徵。

        Returns
        -------
        dict
            center_x, center_y  : 棋盤中心（正規化 0-1）
            region              : 5×5 網格索引 (0-24)（優化 8）
            coverage            : 棋盤覆蓋率
            angle               : 有符號傾斜角度 -90°~+90°（優化 9）
            scale               : 相對尺度
        """
        pts = corners.reshape(-1, 2)
        h, w = image_shape[:2]
        cx, cy = np.mean(pts, axis=0) / [w, h]

        # 5×5 網格分區（優化 8）
        gx = min(int(cx * _GRID_N), _GRID_N - 1)
        gy = min(int(cy * _GRID_N), _GRID_N - 1)
        region = gy * _GRID_N + gx

        x_rng, y_rng = np.max(pts, axis=0) - np.min(pts, axis=0)

        W, H = self.chessboard_size
        top_left = pts[0]
        top_right = pts[W - 1]
        bottom_left = pts[(H - 1) * W]
        bottom_right = pts[H * W - 1]

        # 真正衡量焦距與標定品質的是 out-of-plane tilt (透視變形量)
        top_len = np.linalg.norm(top_right - top_left)
        bottom_len = np.linalg.norm(bottom_right - bottom_left)
        left_len = np.linalg.norm(bottom_left - top_left)
        right_len = np.linalg.norm(bottom_right - top_right)
        
        tilt_x = abs(top_len - bottom_len) / max(top_len, bottom_len, 1e-5)
        tilt_y = abs(left_len - right_len) / max(left_len, right_len, 1e-5)
        tilt = float(tilt_x + tilt_y)

        # 有符號角度，保留平面旋轉方向資訊（優化 9）
        vec = top_right - top_left
        raw_angle = np.degrees(np.arctan2(vec[1], vec[0]))
        # 折疊至 -90°~+90° 保留正負號
        if raw_angle > 90:
            raw_angle -= 180
        elif raw_angle < -90:
            raw_angle += 180
        signed_angle = float(raw_angle)

        return {
            'center_x': float(cx),
            'center_y': float(cy),
            'region': int(region),
            'coverage': float((x_rng * y_rng) / (w * h)),
            'tilt': tilt,
            'angle': signed_angle,
            'scale': float((x_rng / w + y_rng / h) / 2.0),
        }

    def _detect_single(
        self,
        img_path: Path,
        is_left: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        單張影像檢測：提取角點並計算評分。

        Returns
        -------
        dict or None
            None 表示檢測失敗（找不到棋盤格或讀取錯誤）。
        """
        try:
            with open(str(img_path), 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if not ret:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        feats = self.calculate_geometry_features(refined, gray.shape)

        mtx, dist = (self.mtxL, self.distL) if is_left else (self.mtxR, self.distR)
        static_err = self.calculate_static_reprojection_error(refined, mtx, dist)

        # 優化 4：log-scale 正規化評分，避免 error<1 放大問題
        # 修改為使用透視變形量 (tilt) 作為主要獎勵，而非平面旋轉角
        coverage_score = math.log1p(feats['coverage'] * 100)
        tilt = feats['tilt']
        tilt_bonus = min(tilt / 0.15, 1.0)
        tilt_penalty = max(0.0, 1.0 - tilt / 0.03) * 0.5
        
        error_penalty = 1.0 / (1.0 + static_err)
        score = coverage_score * (1.0 + tilt_bonus) * (1.0 - tilt_penalty) * error_penalty

        return {
            'path': str(img_path),
            'filename': img_path.name,
            'corners': refined,
            'image_shape': gray.shape,   # 優化 2：快取影像尺寸，避免重複 I/O
            'score': float(score),
            'static_error': float(static_err),
            **feats,
        }

    # ── 配對與分析 ─────────────────────────────────────────────────────────────

    def analyze_and_pair(
        self,
        left_dir: str,
        right_dir: str,
        logger: Callable[[str], None] = print,
    ) -> int:
        """
        同步配對左右影像並以多執行緒執行初步角點檢測（優化 1）。

        Returns
        -------
        int
            有效配對數量。
        """
        logger("正在分析影像特徵與配對（多執行緒）...")
        l_files = sorted(list(Path(left_dir).glob('*.[jJpP]*')))
        r_map = {p.name: p for p in Path(right_dir).glob('*.[jJpP]*')}

        # 建立 (left_path, right_path) 任務清單
        tasks: List[Tuple[Path, Path]] = [
            (lf, rf)
            for lf in l_files
            if (rf := r_map.get(lf.name)) is not None
        ]

        # 多執行緒並行檢測（優化 1）
        results: Dict[str, Tuple[Optional[Dict], Optional[Dict]]] = {}
        with ThreadPoolExecutor() as executor:
            future_map = {
                executor.submit(self._detect_single, lf, True): (lf, rf, True)
                for lf, rf in tasks
            }
            future_map.update({
                executor.submit(self._detect_single, rf, False): (lf, rf, False)
                for lf, rf in tasks
            })
            raw: Dict[str, Dict[str, Optional[Dict]]] = defaultdict(dict)
            for fut in as_completed(future_map):
                lf, rf, is_left = future_map[fut]
                side = 'left' if is_left else 'right'
                raw[lf.name][side] = fut.result()

        paired: List[Dict[str, Any]] = []
        for lf, _ in tasks:
            entry = raw.get(lf.name, {})
            li, ri = entry.get('left'), entry.get('right')
            if li and ri and li['static_error'] < 10.0 and ri['static_error'] < 10.0:
                # 優化 7：pair_score 改為加權平均並加入左右角度差懲罰
                angle_diff = abs(li['angle'] - ri['angle'])
                angle_diff_penalty = min(angle_diff / 90.0, 1.0) * 0.2
                pair_score = (li['score'] + ri['score']) / 2.0 * (1.0 - angle_diff_penalty)

                paired.append({
                    'pair_key': lf.name,
                    'left': li,
                    'right': ri,
                    'pair_score': float(pair_score),
                    'pair_region': int(li['region']),
                    'static_error_sum': float(li['static_error'] + ri['static_error']),
                })

        self.paired_infos = paired
        logger(f"有效檢測配對數: {len(paired)}")
        return len(paired)

    # ── 標定計算 ───────────────────────────────────────────────────────────────

    def perform_stereo_calibration(
        self,
        selected_pairs: List[Dict[str, Any]],
    ) -> Tuple[float, Any, Any, Any, Any, Any, Any]:
        """執行雙目相機初步標定並返回 RMS 與參數。"""
        if not selected_pairs:
            return float('inf'), None, None, None, None, None, None

        objpoints = [self.objp] * len(selected_pairs)
        imgptsL = [p['left']['corners'] for p in selected_pairs]
        imgptsR = [p['right']['corners'] for p in selected_pairs]

        # 優化 2：直接從快取的 image_shape 取得影像尺寸，不再重複讀檔
        h, w = selected_pairs[0]['left']['image_shape']
        img_size = (w, h)

        try:
            ret, M1, D1, M2, D2, R, T, _, _ = cv2.stereoCalibrate(
                objpoints, imgptsL, imgptsR,
                self.mtxL, self.distL, self.mtxR, self.distR,
                img_size,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
                # flags=cv2.CALIB_USE_INTRINSIC_GUESS
                flags=cv2.CALIB_FIX_INTRINSIC       
            )
            return float(ret), M1, D1, M2, D2, R, T
        except Exception:
            return float('inf'), None, None, None, None, None, None

    # ── 智能挑選主流程 ─────────────────────────────────────────────────────────

    def select_best_pairs(
        self,
        target_rmse: float = 0.5,
        logger: Callable[[str], None] = print,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        雙目智能挑選演算法：
            1. 5×5 空間網格優先填充
            2. 豐富 KMeans 多樣性補充
            3. 帶 Simulated Annealing 的 RMS 疊代優化

        Parameters
        ----------
        target_rmse : float
            目標 RMS 閾值（像素），達到即提前停止。
        logger : Callable
            日誌輸出函式。

        Returns
        -------
        (selected_pairs, final_rms)
        """
        if len(self.paired_infos) < self.target_pairs:
            logger("可用配對不足，回傳全部。")
            return self.paired_infos, 0.0

        logger(f"\n=== 開始智能挑選 (目標: {self.target_pairs} 對，網格: {_GRID_N}×{_GRID_N}) ===")
        candidates = sorted(self.paired_infos, key=lambda x: x['pair_score'], reverse=True)

        # ── Step 1：5×5 空間網格優先填充（優化 8）──────────────────────────────
        selected: List[Dict[str, Any]] = []
        r_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for p in candidates:
            r_map[p['pair_region']].append(p)
        for r in range(_GRID_CELLS):
            if r_map[r]:
                selected.append(r_map[r][0])

        # ── Step 2：豐富 KMeans 多樣性填充（優化 5）──────────────────────────
        needed = self.target_pairs - len(selected)
        if needed > 0:
            pool = [x for x in candidates if x not in selected]
            if pool:
                # 8 維特徵：包含左右畫面的 tilt 與角度，確保多樣性
                feats = np.array([
                    [
                        x['left']['tilt'],
                        x['right']['tilt'],
                        x['left']['angle'],
                        x['right']['angle'],
                        x['left']['scale'],
                        x['right']['scale'],
                        x['left']['center_x'],
                        x['left']['center_y'],
                    ]
                    for x in pool
                ])
                feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-5)
                n_clusters = min(needed, len(pool))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)

                c_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                for i, label in enumerate(kmeans.labels_):
                    c_groups[label].append(pool[i])
                for label in range(n_clusters):
                    if c_groups[label]:
                        selected.append(max(c_groups[label], key=lambda x: x['pair_score']))

            # 數量仍不足時補齊
            while len(selected) < self.target_pairs:
                rem = [x for x in candidates if x not in selected]
                if not rem:
                    break
                selected.append(rem[0])

        selected = selected[:self.target_pairs]
        rms, M1, D1, M2, D2, R, T = self.perform_stereo_calibration(selected)
        logger(f"初始組合 RMS: {rms:.4f} px")

        # ── Step 3：帶 SA 的疊代優化（優化 3、6）──────────────────────────────
        temperature = _SA_INIT_TEMP
        no_improve_count = 0

        for it in range(150):
            if rms <= target_rmse:
                break
            if no_improve_count >= _MAX_NO_IMPROVE:
                logger(f"連續 {_MAX_NO_IMPROVE} 次無改善，提前結束迭代。")
                break

            # 評估各對影像誤差貢獻
            errs: List[float] = []
            for p in selected:
                rL, rvL, tvL = cv2.solvePnP(self.objp, p['left']['corners'], M1, D1)
                rR, rvR, tvR = cv2.solvePnP(self.objp, p['right']['corners'], M2, D2)
                if rL and rR:
                    pL, _ = cv2.projectPoints(self.objp, rvL, tvL, M1, D1)
                    pR, _ = cv2.projectPoints(self.objp, rvR, tvR, M2, D2)
                    err = (
                        cv2.norm(p['left']['corners'], pL, cv2.NORM_L2)
                        + cv2.norm(p['right']['corners'], pR, cv2.NORM_L2)
                    ) / (2.0 * math.sqrt(len(pL)))
                    errs.append(err)
                else:
                    errs.append(float('inf'))

            worst_idx = int(np.argmax(errs))
            worst_p = selected[worst_idx]
            pool = [x for x in candidates if x not in selected]

            # 優先同區域替換，維持空間覆蓋
            repl_pool = (
                [x for x in pool if x['pair_region'] == worst_p['pair_region']] or pool
            )
            if not repl_pool:
                break

            repl = max(repl_pool, key=lambda x: x['pair_score'])
            new_sel = selected.copy()
            new_sel[worst_idx] = repl
            n_rms, nM1, nD1, nM2, nD2, nR, nT = self.perform_stereo_calibration(new_sel)

            delta = n_rms - rms
            # Simulated Annealing：接受較差解的機率（優化 6）
            accept = (delta < 0) or (
                temperature > 1e-6
                and random.random() < math.exp(-delta / temperature)
            )

            if accept:
                action = "改善" if delta < 0 else "SA探索"
                logger(
                    f"Iter {it + 1:3d} [{action}] "
                    f"{worst_p['pair_key']} -> {repl['pair_key']} | "
                    f"RMS: {n_rms:.4f} (Δ{delta:+.4f})"
                )
                selected, rms = new_sel, n_rms
                M1, D1, M2, D2, R, T = nM1, nD1, nM2, nD2, nR, nT
                if delta < 0:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            temperature *= _SA_COOLING

        logger(f"最終 RMS: {rms:.4f} px")

        # 覆蓋度檢查
        final_regions = {p['pair_region'] for p in selected}
        missing = set(range(_GRID_CELLS)) - final_regions
        covered = _GRID_CELLS - len(missing)
        logger(f"空間覆蓋: {covered}/{_GRID_CELLS} 個網格區域" + (
            f"（缺少: {sorted(missing)}）" if missing else "（全部覆蓋）"
        ))

        return selected, rms

    # ── 結果輸出 ───────────────────────────────────────────────────────────────

    def save_results(
        self,
        selected_pairs: List[Dict[str, Any]],
        final_rms: float,
        out_left: str,
        out_right: str,
        logger: Callable[[str], None] = print,  # 優化 15：統一為參數傳遞，不再用 self.logger
    ) -> None:
        """將挑選後的影像配對複製到目標資料夾並產出 JSON 報告與分析圖。"""
        logger("\n=== 開始輸出結果 ===")
        os.makedirs(out_left, exist_ok=True)
        os.makedirs(out_right, exist_ok=True)

        summary: List[Dict[str, Any]] = []
        for i, p in enumerate(selected_pairs):
            try:
                fn_l = f"stereo_{i + 1:02d}_L_{p['left']['filename']}"
                fn_r = f"stereo_{i + 1:02d}_R_{p['right']['filename']}"

                for src, dst in [
                    (p['left']['path'], os.path.join(out_left, fn_l)),
                    (p['right']['path'], os.path.join(out_right, fn_r)),
                ]:
                    with open(src, 'rb') as f:
                        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        _, enc = cv2.imencode('.jpg', img)
                        with open(dst, 'wb') as f:
                            f.write(enc.tobytes())

                summary.append({
                    'id': i + 1,
                    'file': p['pair_key'],
                    'region': p['pair_region'],
                    'score': round(p['pair_score'], 4),
                    'err': round(p['static_error_sum'], 4),
                })
            except Exception as e:
                logger(f"儲存失敗 (ID {i + 1}): {e}")

        # 輸出 JSON 報表
        report_dir = Path(out_left).parent.parent
        report_path = report_dir / 'calibration_selection_report.json'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {'total': len(selected_pairs), 'rms': round(final_rms, 4), 'details': summary},
                    f, indent=2, ensure_ascii=False,
                )
        except Exception as e:
            logger(f"JSON 報表寫入失敗: {e}")

        # 繪製分析圖
        try:
            self.plot_analysis(selected_pairs, final_rms, report_dir, logger)
        except Exception as e:
            logger(f"分析圖繪製失敗: {e}")

    def plot_analysis(
        self,
        pairs: List[Dict[str, Any]],
        rms: float,
        out_dir: Path,
        logger: Callable[[str], None] = print,
    ) -> None:
        """繪製分析圖表：5×5 空間熱圖與有符號角度分佈（優化 8、9）。"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 5×5 空間熱圖
        grid = np.zeros((_GRID_N, _GRID_N), dtype=int)
        for p in pairs:
            r = p['pair_region']
            grid[r // _GRID_N, r % _GRID_N] += 1
        im = axes[0].imshow(grid, cmap='Blues', aspect='auto')
        axes[0].set_title(f'空間分佈熱圖 ({_GRID_N}×{_GRID_N} 網格)')
        axes[0].set_xlabel('X 網格')
        axes[0].set_ylabel('Y 網格')
        plt.colorbar(im, ax=axes[0])

        # 有符號角度分佈直方圖
        angles = [p['left']['angle'] for p in pairs]
        axes[1].hist(angles, bins=15, color='salmon', alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[1].set_title(f'傾斜角度分佈 (RMS: {rms:.4f} px)')
        axes[1].set_xlabel('角度 (度)')
        axes[1].set_ylabel('數量')

        plt.tight_layout()
        save_path = os.path.join(out_dir, 'selection_analysis.png')
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        logger(f"分析圖已儲存: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 公開介面
# ═══════════════════════════════════════════════════════════════════════════════

def run_stereo_pick(
    left_dir: str,
    right_dir: str,
    out_left: str,
    out_right: str,
    mtxL: np.ndarray,
    distL: np.ndarray,
    mtxR: np.ndarray,
    distR: np.ndarray,
    chessboard_size: Tuple[int, int] = (9, 6),
    target_pairs: int = 15,
    logger: Callable[[str], None] = print,
    square_size: float = 0.09,   # 優化 14：由外部傳入
) -> Tuple[int, float]:
    """
    雙目影像挑選主進入點。

    Returns
    -------
    (selected_count, final_rms)
    """
    sel = OptimizedStereoChessboardSelector(
        mtxL, distL, mtxR, distR,
        chessboard_size=chessboard_size,
        target_pairs=target_pairs,
        square_size=square_size,
    )
    if sel.analyze_and_pair(left_dir, right_dir, logger) == 0:
        return 0, 0.0
    selected, rms = sel.select_best_pairs(target_rmse=0.3, logger=logger)
    sel.save_results(selected, rms, out_left, out_right, logger)
    return len(selected), rms