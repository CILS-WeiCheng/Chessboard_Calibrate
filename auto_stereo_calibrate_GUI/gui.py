import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os
import sys
import numpy as np
from pathlib import Path

# 加入路徑以支援 utils 讀取
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import video_processor, stereo_picker, calibrator

class AutoStereoCalibrateApp:
    """雙目自動標定系統介面程式"""
    def __init__(self, root):
        self.root = root
        self.root.title("自動雙目標定系統")
        self.root.geometry("700x900")
        self.is_running = False
        self.label_font, self.entry_font = ('Microsoft JhengHei', 10), ('Arial', 10)
        self.MODE_FULL, self.MODE_DIRECT = "FULL", "DIRECT"
        self.mode_var = tk.StringVar(value=self.MODE_FULL)
        self.create_widgets()
        
    def create_widgets(self):
        """初始化介面組件"""
        # 0. 模式切換
        frame_mode = tk.LabelFrame(self.root, text="工作模式", font=self.label_font, padx=10, pady=5)
        frame_mode.pack(fill="x", padx=10, pady=5)
        for t, v in [("完整流程 (影片)", self.MODE_FULL), ("快速標定 (資料夾)", self.MODE_DIRECT)]:
            tk.Radiobutton(frame_mode, text=t, variable=self.mode_var, value=v, 
                           command=self.toggle_mode, font=self.label_font).pack(side="left", padx=10)

        # 1. 動態輸入區
        self.cont = tk.Frame(self.root); self.cont.pack(fill="x")
        
        self.f_v = tk.LabelFrame(self.cont, text="1. 輸入影片 (L/R)", font=self.label_font, padx=10, pady=10)
        self.f_v.pack(fill="x", padx=10, pady=5)
        self.e_vL = self._add_row(self.f_v, "左視角影片:", 0, "Video")
        self.e_vR = self._add_row(self.f_v, "右視角影片:", 1, "Video")

        self.f_d = tk.LabelFrame(self.cont, text="1. 輸入圖片資料夾 (L/R)", font=self.label_font, padx=10, pady=10)
        self.e_dL = self._add_row(self.f_d, "左圖片目錄:", 0, "Dir")
        self.e_dR = self._add_row(self.f_d, "右圖片目錄:", 1, "Dir")

        # 2. 輸出與內參
        f_o = tk.LabelFrame(self.root, text="2. 輸出與內參", font=self.label_font, padx=10, pady=10)
        f_o.pack(fill="x", padx=10, pady=5)
        self.e_out = self._add_row(f_o, "工作目錄:", 0, "Dir")
        self.e_nL = self._add_row(f_o, "左內參 NPZ:", 1, "NPZ")
        self.e_nR = self._add_row(f_o, "右內參 NPZ:", 2, "NPZ")

        # 3. 參數
        f_p = tk.LabelFrame(self.root, text="3. 參數設定", font=self.label_font, padx=10, pady=10)
        f_p.pack(fill="x", padx=10, pady=5)
        self.e_int = self._add_param(f_p, "切幀間隔:", 0, 0, "5")
        self.e_sq = self._add_param(f_p, "方格(m):", 1, 0, "0.09")
        self.e_cnt = self._add_param(f_p, "目標組數:", 1, 2, "15")
        
        tk.Label(f_p, text="棋盤尺寸:", font=self.label_font).grid(row=0, column=2, sticky="w", padx=5)
        f_s = tk.Frame(f_p); f_s.grid(row=0, column=3, sticky="w")
        self.e_col = tk.Entry(f_s, width=4); self.e_col.insert(0, "9"); self.e_col.pack(side="left")
        tk.Label(f_s, text="x").pack(side="left")
        self.e_row = tk.Entry(f_s, width=4); self.e_row.insert(0, "6"); self.e_row.pack(side="left")

        # 4. 控制與日誌
        self.btn_run = tk.Button(self.root, text="開始自動標定", command=self.start_process, bg="#4CAF50", fg="white", font=('Microsoft JhengHei', 12, 'bold'), height=2)
        self.btn_run.pack(fill="x", padx=20, pady=10)
        self.log_a = scrolledtext.ScrolledText(self.root, height=12, font=('Consolas', 9)); self.log_a.pack(fill="both", expand=True, padx=10, pady=5)

    def _add_row(self, master, label, row, type):
        tk.Label(master, text=label, font=self.label_font).grid(row=row, column=0, sticky="w")
        e = tk.Entry(master, width=45, font=self.entry_font); e.grid(row=row, column=1, padx=5, pady=2)
        cmd = self.browse_dir if type == "Dir" else lambda: self.browse_file(e, type)
        tk.Button(master, text="瀏覽", command=cmd if type != "Dir" else lambda: self.browse_dir(e), font=self.label_font).grid(row=row, column=2)
        return e

    def _add_param(self, master, label, row, col, val):
        tk.Label(master, text=label, font=self.label_font).grid(row=row, column=col, sticky="w", padx=5)
        e = tk.Entry(master, width=8); e.insert(0, val); e.grid(row=row, column=col+1, sticky="w", pady=2)
        return e

    def toggle_mode(self):
        """切換輸入模式介面"""
        if self.mode_var.get() == self.MODE_FULL:
            self.f_d.pack_forget(); self.f_v.pack(fill="x", padx=10, pady=5)
        else:
            self.f_v.pack_forget(); self.f_d.pack(fill="x", padx=10, pady=5)

    def browse_dir(self, e):
        d = filedialog.askdirectory()
        if d: e.delete(0, tk.END); e.insert(0, d)

    def browse_file(self, e, t):
        ft = [("Video", "*.mp4 *.avi *.mov *.mkv"), ("NPZ", "*.npz")] if t == "Video" else [("NPZ", "*.npz")]
        f = filedialog.askopenfilename(filetypes=ft)
        if f: e.delete(0, tk.END); e.insert(0, f)

    def log(self, msg):
        self.log_a.insert(tk.END, msg + "\n"); self.log_a.see(tk.END); print(msg)

    def load_npz(self, path):
        """讀取內參 NPZ 檔案"""
        try:
            d = np.load(path)
            mtx = d['camera_matrix'] if 'camera_matrix' in d else d.get('mtx')
            dist = d['dist_coeffs'] if 'dist_coeffs' in d else d.get('dist')
            if mtx is None or dist is None: return None, None, "格式不符"
            return mtx, dist, None
        except Exception as e: return None, None, str(e)

    def start_process(self):
        """啟動非同步流水線"""
        if self.is_running: return
        mode = self.mode_var.get()
        p = {
            'mode': mode, 'out': self.e_out.get(), 'nL': self.e_nL.get(), 'nR': self.e_nR.get(),
            'vL': self.e_vL.get(), 'vR': self.e_vR.get(), 'dL': self.e_dL.get(), 'dR': self.e_dR.get(),
            'interval': int(self.e_int.get()), 'square': float(self.e_sq.get()), 'count': int(self.e_cnt.get()),
            'size': (int(self.e_col.get()), int(self.e_row.get()))
        }
        if not p['out'] or not os.path.exists(p['nL']): messagebox.showerror("錯誤", "請檢查路徑設定"); return
        
        self.is_running = True; self.btn_run.config(state="disabled", text="執行中...")
        self.log_a.delete(1.0, tk.END)
        threading.Thread(target=self.run_pipeline, kwargs=p).start()

    def run_pipeline(self, mode, out, nL, nR, vL, vR, dL, dR, interval, square, count, size):
        """核心自動化流水線邏輯"""
        try:
            self.log("Step 0: 讀取內參...")
            mL, dL_o, eL = self.load_npz(nL); mR, dR_o, eR = self.load_npz(nR)
            if eL or eR: raise Exception(f"內參讀取失敗: {eL or eR}")

            if mode == self.MODE_FULL:
                dirs = {k: os.path.join(out, "stereo", side, k) for side in ["left", "right"] for k in ["origin_image", "final_image"]}
                self.log("\nStep 1: 提取影片幀..."); video_processor.extract_frames(vL, dirs["leftorigin_image"], interval, self.log)
                video_processor.extract_frames(vR, dirs["rightorigin_image"], interval, self.log)
                
                self.log(f"\nStep 3: 智慧挑選影像對 (目標: {count})..."); stereo_picker.run_stereo_pick(
                    dirs["leftorigin_image"], dirs["rightorigin_image"], dirs["leftfinal_image"], dirs["rightfinal_image"],
                    mL, dL_o, mR, dR_o, size, count, self.log)
                fL, fR = dirs["leftfinal_image"], dirs["rightfinal_image"]
            else:
                fL, fR = dL, dR

            self.log("\nStep 5: 執行雙目最終標定...")
            res = calibrator.stereo_calibration(fL, fR, mL, dL_o, mR, dR_o, os.path.join(out, "stereo_rt.npz"), size, square, self.log)
            if res: messagebox.showinfo("完成", f"標定成功！RMS: {res['ret']:.6f}\nBaseline: {res['baseline']:.4f}m")
        except Exception as e: self.log(f"\n[嚴重錯誤]: {e}"); messagebox.showerror("錯誤", str(e))
        finally: self.is_running = False; self.btn_run.config(state="normal", text="開始自動標定")

if __name__ == "__main__":
    root = tk.Tk(); app = AutoStereoCalibrateApp(root); root.mainloop()
