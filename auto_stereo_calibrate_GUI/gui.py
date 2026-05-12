"""
gui.py
======
自動雙目標定系統介面程式。

優化歷程：
    四、GUI 穩健性 - 完整路徑驗證、traceback 日誌、ttk.Progressbar、_add_row 修正
    五、可維護性   - square_size 傳遞至 run_stereo_pick
"""

import os
import sys
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path

# 加入路徑以支援 utils 讀取
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import video_processor, stereo_picker, calibrator


class AutoStereoCalibrateApp:
    """雙目自動標定系統介面程式。"""

    # ── 工作模式常數 ──────────────────────────────────────────────────────────
    MODE_FULL = "FULL"      # 完整流程（影片 -> 取幀 -> 挑選 -> 標定）
    MODE_PICK = "PICK"      # 智慧挑選 + 標定（圖片資料夾）
    MODE_DIRECT = "DIRECT"  # 快速標定（直接使用現有圖片）

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("自動雙目標定系統")
        self.root.geometry("700x960")
        self.is_running = False
        self.label_font = ('Microsoft JhengHei', 10)
        self.entry_font = ('Arial', 10)
        self.mode_var = tk.StringVar(value=self.MODE_FULL)
        self._create_widgets()

    # ── 介面建立 ───────────────────────────────────────────────────────────────

    def _create_widgets(self) -> None:
        """初始化所有介面組件。"""
        # 0. 模式切換
        frame_mode = tk.LabelFrame(
            self.root, text="工作模式", font=self.label_font, padx=10, pady=5
        )
        frame_mode.pack(fill="x", padx=10, pady=5)
        modes = [
            ("完整流程 (影片)", self.MODE_FULL),
            ("智慧挑選+標定 (資料夾)", self.MODE_PICK),
            ("快速標定 (僅標定)", self.MODE_DIRECT),
        ]
        for text, value in modes:
            tk.Radiobutton(
                frame_mode, text=text, variable=self.mode_var, value=value,
                command=self._toggle_mode, font=self.label_font,
            ).pack(side="left", padx=10)

        # 1. 動態輸入區（影片 / 圖片資料夾）
        self.cont = tk.Frame(self.root)
        self.cont.pack(fill="x")

        self.f_v = tk.LabelFrame(
            self.cont, text="1. 輸入影片 (L/R)", font=self.label_font, padx=10, pady=10
        )
        self.f_v.pack(fill="x", padx=10, pady=5)
        self.e_vL = self._add_row(self.f_v, "左視角影片:", 0, "Video")
        self.e_vR = self._add_row(self.f_v, "右視角影片:", 1, "Video")

        self.f_d = tk.LabelFrame(
            self.cont, text="1. 輸入圖片資料夾 (L/R)", font=self.label_font, padx=10, pady=10
        )
        self.e_dL = self._add_row(self.f_d, "左圖片目錄:", 0, "Dir")
        self.e_dR = self._add_row(self.f_d, "右圖片目錄:", 1, "Dir")

        # 2. 輸出與內參
        f_o = tk.LabelFrame(
            self.root, text="2. 輸出與內參", font=self.label_font, padx=10, pady=10
        )
        f_o.pack(fill="x", padx=10, pady=5)
        self.e_out = self._add_row(f_o, "工作目錄:", 0, "Dir")
        self.e_nL = self._add_row(f_o, "左內參 NPZ:", 1, "NPZ")
        self.e_nR = self._add_row(f_o, "右內參 NPZ:", 2, "NPZ")

        # 3. 參數設定
        f_p = tk.LabelFrame(
            self.root, text="3. 參數設定", font=self.label_font, padx=10, pady=10
        )
        f_p.pack(fill="x", padx=10, pady=5)
        self.e_int = self._add_param(f_p, "切幀間隔:", 0, 0, "5")
        self.e_sq = self._add_param(f_p, "方格(m):", 1, 0, "0.09")
        self.e_cnt = self._add_param(f_p, "目標組數:", 1, 2, "15")

        tk.Label(f_p, text="棋盤尺寸:", font=self.label_font).grid(
            row=0, column=2, sticky="w", padx=5
        )
        f_s = tk.Frame(f_p)
        f_s.grid(row=0, column=3, sticky="w")
        self.e_col = tk.Entry(f_s, width=4)
        self.e_col.insert(0, "9")
        self.e_col.pack(side="left")
        tk.Label(f_s, text="x").pack(side="left")
        self.e_row = tk.Entry(f_s, width=4)
        self.e_row.insert(0, "6")
        self.e_row.pack(side="left")

        # 4. 執行按鈕
        self.btn_run = tk.Button(
            self.root, text="開始自動標定",
            command=self._start_process,
            bg="#4CAF50", fg="white",
            font=('Microsoft JhengHei', 12, 'bold'),
            height=2,
        )
        self.btn_run.pack(fill="x", padx=20, pady=10)

        # 5. 進度條（優化 12）
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.root, variable=self.progress_var,
            maximum=100.0, mode='determinate',
        )
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 5))

        # 6. 日誌視窗
        self.log_area = scrolledtext.ScrolledText(
            self.root, height=14, font=('Consolas', 9)
        )
        self.log_area.pack(fill="both", expand=True, padx=10, pady=5)

    def _add_row(self, master: tk.Widget, label: str, row: int, kind: str) -> tk.Entry:
        """
        在 master 的 Grid 中新增一列（標籤 + 輸入框 + 瀏覽按鈕）。

        Parameters
        ----------
        kind : str
            "Dir" | "Video" | "NPZ"
        """
        tk.Label(master, text=label, font=self.label_font).grid(
            row=row, column=0, sticky="w"
        )
        entry = tk.Entry(master, width=45, font=self.entry_font)
        entry.grid(row=row, column=1, padx=5, pady=2)

        # 優化 13：清晰 if/else，不再有冗餘 cmd 賦值
        if kind == "Dir":
            cmd = lambda e=entry: self._browse_dir(e)
        else:
            cmd = lambda e=entry, k=kind: self._browse_file(e, k)

        tk.Button(master, text="瀏覽", command=cmd, font=self.label_font).grid(
            row=row, column=2
        )
        return entry

    def _add_param(
        self, master: tk.Widget, label: str, row: int, col: int, val: str
    ) -> tk.Entry:
        """在 master 的 Grid 中新增參數輸入列。"""
        tk.Label(master, text=label, font=self.label_font).grid(
            row=row, column=col, sticky="w", padx=5
        )
        entry = tk.Entry(master, width=8)
        entry.insert(0, val)
        entry.grid(row=row, column=col + 1, sticky="w", pady=2)
        return entry

    # ── 介面事件 ───────────────────────────────────────────────────────────────

    def _toggle_mode(self) -> None:
        """切換輸入模式介面（影片區 ↔ 圖片資料夾區）。"""
        if self.mode_var.get() == self.MODE_FULL:
            self.f_d.pack_forget()
            self.f_v.pack(fill="x", padx=10, pady=5)
        else:
            self.f_v.pack_forget()
            self.f_d.pack(fill="x", padx=10, pady=5)

    def _browse_dir(self, entry: tk.Entry) -> None:
        """開啟資料夾選擇對話框並填入 entry。"""
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def _browse_file(self, entry: tk.Entry, kind: str) -> None:
        """開啟檔案選擇對話框並填入 entry。"""
        filetypes = (
            [("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
            if kind == "Video"
            else [("NPZ", "*.npz"), ("All", "*.*")]
        )
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    # ── 日誌與進度 ─────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        """將訊息寫入日誌視窗（執行緒安全）。"""
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        print(msg)

    def _set_progress(self, value: float) -> None:
        """更新進度條（0-100），可由子執行緒呼叫（優化 12）。"""
        self.progress_var.set(value)
        self.root.update_idletasks()

    # ── 內參讀取 ───────────────────────────────────────────────────────────────

    def _load_npz(self, path: str):
        """
        讀取內參 NPZ 檔案。

        Returns
        -------
        (mtx, dist, error_msg)
            error_msg 為 None 表示成功。
        """
        try:
            data = np.load(path)
            mtx = data.get('camera_matrix') if 'camera_matrix' in data else data.get('mtx')
            dist = data.get('dist_coeffs') if 'dist_coeffs' in data else data.get('dist')
            if mtx is None or dist is None:
                return None, None, "格式不符（缺少 camera_matrix/mtx 或 dist_coeffs/dist）"
            return mtx, dist, None
        except Exception as e:
            return None, None, str(e)

    # ── 流程啟動 ───────────────────────────────────────────────────────────────

    def _validate_params(self, mode: str, params: dict) -> str:
        """
        依模式驗證所有必要路徑與參數（優化 10）。

        Returns
        -------
        str
            空字串表示驗證通過；否則為錯誤訊息。
        """
        if not params['out']:
            return "請設定工作目錄。"
        if not params['out'] or not os.path.isdir(params['out']):
            return f"工作目錄不存在：{params['out']}"
        if not os.path.isfile(params['nL']):
            return f"左內參 NPZ 不存在：{params['nL']}"
        if not os.path.isfile(params['nR']):
            return f"右內參 NPZ 不存在：{params['nR']}"

        if mode == self.MODE_FULL:
            if not os.path.isfile(params['vL']):
                return f"左視角影片不存在：{params['vL']}"
            if not os.path.isfile(params['vR']):
                return f"右視角影片不存在：{params['vR']}"
        elif mode in (self.MODE_PICK, self.MODE_DIRECT):
            if not os.path.isdir(params['dL']):
                return f"左圖片目錄不存在：{params['dL']}"
            if not os.path.isdir(params['dR']):
                return f"右圖片目錄不存在：{params['dR']}"

        return ""

    def _start_process(self) -> None:
        """收集參數並啟動非同步流水線。"""
        if self.is_running:
            return

        mode = self.mode_var.get()
        params = {
            'mode': mode,
            'out': self.e_out.get().strip(),
            'nL': self.e_nL.get().strip(),
            'nR': self.e_nR.get().strip(),
            'vL': self.e_vL.get().strip(),
            'vR': self.e_vR.get().strip(),
            'dL': self.e_dL.get().strip(),
            'dR': self.e_dR.get().strip(),
            'interval': int(self.e_int.get()),
            'square': float(self.e_sq.get()),
            'count': int(self.e_cnt.get()),
            'size': (int(self.e_col.get()), int(self.e_row.get())),
        }

        # 完整路徑驗證（優化 10）
        err_msg = self._validate_params(mode, params)
        if err_msg:
            messagebox.showerror("輸入錯誤", err_msg)
            return

        self.is_running = True
        self.btn_run.config(state="disabled", text="執行中...")
        self.log_area.delete(1.0, tk.END)
        self._set_progress(0.0)

        threading.Thread(target=self._run_pipeline, kwargs=params, daemon=True).start()

    # ── 核心流水線 ─────────────────────────────────────────────────────────────

    def _run_pipeline(
        self, mode, out, nL, nR, vL, vR, dL, dR, interval, square, count, size
    ) -> None:
        """核心自動化流水線邏輯（在子執行緒中執行）。"""
        try:
            # Step 0：讀取內參
            self._log("Step 0: 讀取內參...")
            self._set_progress(5.0)
            mL, dL_o, eL = self._load_npz(nL)
            mR, dR_o, eR = self._load_npz(nR)
            if eL or eR:
                raise ValueError(f"內參讀取失敗: {eL or eR}")

            if mode == self.MODE_FULL:
                # Step 1：提取影片幀
                dirs = {
                    f"{side}_{k}": os.path.join(out, "stereo", side, k)
                    for side in ["left", "right"]
                    for k in ["origin_image", "final_image"]
                }
                self._log("\nStep 1: 提取左影片幀...")
                self._set_progress(10.0)
                video_processor.extract_frames(
                    vL, dirs["left_origin_image"], interval, self._log
                )
                self._log("Step 1: 提取右影片幀...")
                self._set_progress(20.0)
                video_processor.extract_frames(
                    vR, dirs["right_origin_image"], interval, self._log
                )

                # Step 2：智慧挑選
                self._log(f"\nStep 2: 智慧挑選影像對 (目標: {count})...")
                self._set_progress(35.0)
                stereo_picker.run_stereo_pick(
                    dirs["left_origin_image"], dirs["right_origin_image"],
                    dirs["left_final_image"], dirs["right_final_image"],
                    mL, dL_o, mR, dR_o,
                    chessboard_size=size, target_pairs=count,
                    logger=self._log, square_size=square,  # 優化 14
                )
                fL, fR = dirs["left_final_image"], dirs["right_final_image"]

            elif mode == self.MODE_PICK:
                dirs = {
                    f"{side}_final_image": os.path.join(out, "stereo", side, "final_image")
                    for side in ["left", "right"]
                }
                self._log(f"\nStep 2: 智慧挑選影像對 (目標: {count})...")
                self._set_progress(35.0)
                stereo_picker.run_stereo_pick(
                    dL, dR,
                    dirs["left_final_image"], dirs["right_final_image"],
                    mL, dL_o, mR, dR_o,
                    chessboard_size=size, target_pairs=count,
                    logger=self._log, square_size=square,  # 優化 14
                )
                fL, fR = dirs["left_final_image"], dirs["right_final_image"]

            else:
                # MODE_DIRECT：直接使用指定資料夾
                fL, fR = dL, dR

            # Step 3：雙目最終標定
            self._log("\nStep 3: 執行雙目最終標定...")
            self._set_progress(70.0)
            res = calibrator.stereo_calibration(
                fL, fR, mL, dL_o, mR, dR_o,
                save_path=os.path.join(out, "stereo_rt.npz"),
                chessboard_size=size,
                square_size=square,
                logger=self._log,
            )
            self._set_progress(100.0)

            if res:
                messagebox.showinfo(
                    "完成",
                    f"標定成功！\nRMS: {res['ret']:.6f}\nBaseline: {res['baseline']:.4f} m",
                )

        except Exception as e:
            # 優化 11：將完整 traceback 寫入日誌
            tb = traceback.format_exc()
            self._log(f"\n[嚴重錯誤]: {e}\n{tb}")
            messagebox.showerror("錯誤", str(e))

        finally:
            self.is_running = False
            self.btn_run.config(state="normal", text="開始自動標定")


if __name__ == "__main__":
    root = tk.Tk()
    app = AutoStereoCalibrateApp(root)
    root.mainloop()
