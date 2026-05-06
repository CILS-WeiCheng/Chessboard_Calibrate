import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os
import sys
import numpy as np
from pathlib import Path

# 加入當前目錄以確保 utils 可正確匯入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import video_processor, auto_picker, calibrator

class AutoCalibrateApp:
    """
    單目相機自動標定 GUI 應用程式
    功能：自動化處理「影片切幀 -> 棋盤格初篩 -> 智能影像挑選 -> 最終標定產出」的完整工作流。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("棋盤格自動標定")
        self.root.geometry("600x850")
        
        self.is_running = False
        
        # Styles
        self.label_font = ('Microsoft JhengHei', 10)
        self.entry_font = ('Arial', 10)
        
        # Modes
        self.MODE_VIDEO = "1. 從影片開始 (完整流程)"
        self.MODE_ORIGIN = "2. 從原始圖片 (origin_image) 開始"
        self.MODE_BEST = "3. 從篩選後圖片 (best_image) 開始"
        self.MODE_FINAL = "4. 從最終圖片 (final_img) 開始標定"
        
        self.create_widgets()
        
    def create_widgets(self):
        """初始化 UI 組件"""
        # 1. 模式選擇
        frame_mode = tk.LabelFrame(self.root, text="執行模式", font=self.label_font, padx=10, pady=10)
        frame_mode.pack(fill="x", padx=10, pady=5)
        
        self.combo_mode = ttk.Combobox(frame_mode, values=[self.MODE_VIDEO, self.MODE_ORIGIN, self.MODE_BEST, self.MODE_FINAL], 
                                      state="readonly", font=self.label_font, width=40)
        self.combo_mode.current(0)
        self.combo_mode.pack(side="left", padx=5)
        self.combo_mode.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Input Section
        frame_input = tk.LabelFrame(self.root, text="路徑設定", font=self.label_font, padx=10, pady=10)
        frame_input.pack(fill="x", padx=10, pady=5)
        
        self.lbl_input = tk.Label(frame_input, text="影片路徑:", font=self.label_font)
        self.lbl_input.grid(row=0, column=0, sticky="w")
        
        self.entry_input = tk.Entry(frame_input, width=50, font=self.entry_font)
        self.entry_input.grid(row=0, column=1, padx=5)
        
        self.btn_browse_input = tk.Button(frame_input, text="瀏覽", command=self.browse_input, font=self.label_font)
        self.btn_browse_input.grid(row=0, column=2)
        
        tk.Label(frame_input, text="輸出/工作目錄:", font=self.label_font).grid(row=1, column=0, sticky="w")
        self.entry_output = tk.Entry(frame_input, width=50, font=self.entry_font)
        self.entry_output.grid(row=1, column=1, padx=5)
        tk.Button(frame_input, text="瀏覽", command=self.browse_output, font=self.label_font).grid(row=1, column=2)

        # Parameters Section
        frame_params = tk.LabelFrame(self.root, text="參數設定", font=self.label_font, padx=10, pady=10)
        frame_params.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_params, text="切幀間隔 (Frame Extractor):", font=self.label_font).grid(row=0, column=0, sticky="w")
        self.entry_interval = tk.Entry(frame_params, width=10, font=self.entry_font)
        self.entry_interval.insert(0, "5")
        self.entry_interval.grid(row=0, column=1, sticky="w")
        
        tk.Label(frame_params, text="棋盤格尺寸 (Cols, Rows):", font=self.label_font).grid(row=1, column=0, sticky="w")
        frame_size = tk.Frame(frame_params)
        frame_size.grid(row=1, column=1, sticky="w")
        self.entry_cols = tk.Entry(frame_size, width=5, font=self.entry_font)
        self.entry_cols.insert(0, "9")
        self.entry_cols.pack(side="left")
        tk.Label(frame_size, text="x").pack(side="left")
        self.entry_rows = tk.Entry(frame_size, width=5, font=self.entry_font)
        self.entry_rows.insert(0, "6")
        self.entry_rows.pack(side="left")

        tk.Label(frame_params, text="方格大小 (m):", font=self.label_font).grid(row=2, column=0, sticky="w")
        self.entry_square = tk.Entry(frame_params, width=10, font=self.entry_font)
        self.entry_square.insert(0, "0.09") 
        self.entry_square.grid(row=2, column=1, sticky="w")
        
        tk.Label(frame_params, text="目標圖片數量:", font=self.label_font).grid(row=3, column=0, sticky="w")
        self.entry_count = tk.Entry(frame_params, width=10, font=self.entry_font)
        self.entry_count.insert(0, "15")
        self.entry_count.grid(row=3, column=1, sticky="w")

        # Action Section
        self.btn_start = tk.Button(self.root, text="開始執行", command=self.start_process, 
                                   bg="#4CAF50", fg="white", font=('Microsoft JhengHei', 12, 'bold'), height=2)
        self.btn_start.pack(fill="x", padx=20, pady=10)
        
        # Tools Section
        frame_tools = tk.LabelFrame(self.root, text="工具", font=self.label_font, padx=10, pady=5)
        frame_tools.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_tools, text="NPZ 檔案路徑:", font=self.label_font).grid(row=0, column=0, sticky="w")
        self.entry_npz = tk.Entry(frame_tools, width=50, font=self.entry_font)
        self.entry_npz.grid(row=0, column=1, padx=5)
        tk.Button(frame_tools, text="瀏覽", command=self.browse_npz, font=self.label_font).grid(row=0, column=2)
        
        tk.Button(frame_tools, text="檢視標定結果", command=self.inspect_npz, 
                  font=self.label_font, bg="#2196F3", fg="white").grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Log Section
        tk.Label(self.root, text="執行紀錄:", font=self.label_font).pack(anchor="w", padx=10)
        self.log_area = scrolledtext.ScrolledText(self.root, height=12, font=('Consolas', 9))
        self.log_area.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def on_mode_change(self, event):
        """當切換模式時，動態調整輸入提示"""
        mode = self.combo_mode.get()
        self.entry_input.delete(0, tk.END)
        mapping = {self.MODE_VIDEO: "影片路徑:", self.MODE_ORIGIN: "原始圖片資料夾 (origin_image):",
                   self.MODE_BEST: "初篩後圖片資料夾 (best_image):", self.MODE_FINAL: "最終圖片資料夾 (final_img):"}
        self.lbl_input.config(text=mapping.get(mode, "路徑:"))
        self.entry_interval.config(state="normal" if mode == self.MODE_VIDEO else "disabled")

    def browse_input(self):
        mode = self.combo_mode.get()
        if mode == self.MODE_VIDEO:
            path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        else: path = filedialog.askdirectory()
        if path: self.entry_input.delete(0, tk.END); self.entry_input.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path: self.entry_output.delete(0, tk.END); self.entry_output.insert(0, path)

    def browse_npz(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ", "*.npz")])
        if path: self.entry_npz.delete(0, tk.END); self.entry_npz.insert(0, path)
            
    def log(self, message):
        """日誌輸出至介面與終端機"""
        self.log_area.insert(tk.END, message + "\n"); self.log_area.see(tk.END)
        print(message)
        
    def inspect_npz(self):
        """解構並顯示 .npz 檔案內容"""
        fn = self.entry_npz.get()
        if not fn or not os.path.exists(fn): messagebox.showerror("錯誤", "無效的 NPZ 檔案"); return
        try:
            data = np.load(fn, allow_pickle=True)
            info = [f"檔案: {os.path.basename(fn)}", "-"*30]
            for k in data.files:
                v = data[k]
                info.append(f"Key: {k}\n  Shape: {v.shape if hasattr(v, 'shape') else 'N/A'}\n  Value: {v if v.size < 15 else '...大型矩陣...'}\n")
            top = tk.Toplevel(self.root); top.title("NPZ 內容")
            txt = scrolledtext.ScrolledText(top, font=('Consolas', 11)); txt.pack(fill="both", expand=True)
            txt.insert(tk.END, "\n".join(info))
        except Exception as e: messagebox.showerror("錯誤", f"讀取失敗: {e}")

    def start_process(self):
        """啟動非同步流水線執行"""
        if self.is_running: return
        try:
            params = {
                'mode': self.combo_mode.get(), 'input_path': self.entry_input.get(), 'output_base': self.entry_output.get(),
                'interval': int(self.entry_interval.get()), 'chessboard_size': (int(self.entry_cols.get()), int(self.entry_rows.get())),
                'square_size': float(self.entry_square.get()), 'target_count': int(self.entry_count.get())
            }
        except ValueError: messagebox.showerror("錯誤", "參數格式不正確"); return
            
        if not params['output_base'] or not os.path.exists(params['input_path']):
            messagebox.showerror("錯誤", "路徑無效"); return

        self.is_running = True
        self.btn_start.config(state="disabled", text="執行中...")
        self.log_area.delete(1.0, tk.END)
        threading.Thread(target=self.run_pipeline, kwargs=params).start()
        
    def run_pipeline(self, mode, input_path, output_base, interval, chessboard_size, square_size, target_count):
        """核心自動化流水線邏輯"""
        try:
            dirs = {k: os.path.join(output_base, k) for k in ["origin_image", "best_image", "final_img"]}
            
            # Step 1: 影片切幀
            cur_dir = input_path
            if mode == self.MODE_VIDEO:
                self.log("Step 1: 提取影片幀..."); video_processor.extract_frames(input_path, dirs["origin_image"], interval, self.log)
                cur_dir = dirs["origin_image"]
            
            # Step 2 & 3: 初步篩選
            if mode in [self.MODE_VIDEO, self.MODE_ORIGIN]:
                self.log(f"\nStep 2 & 3: 篩選棋盤格影像 (來源: {cur_dir})...")
                video_processor.filter_valid_images(cur_dir, dirs["best_image"], chessboard_size, self.log)
                cur_dir = dirs["best_image"]

            # Step 4: 智能挑選
            if mode in [self.MODE_VIDEO, self.MODE_ORIGIN, self.MODE_BEST]:
                self.log(f"\nStep 4: 執行智慧挑選 (目標: {target_count} 張)...")
                auto_picker.run_auto_pick(cur_dir, dirs["final_img"], chessboard_size, square_size, target_count, self.log)
                cur_dir = dirs["final_img"]

            # Step 5 & 6: 標定產出
            self.log("\nStep 5 & 6: 執行最終標定..."); npz_path = os.path.join(output_base, "calibration_result.npz")
            res = calibrator.calibration_final(os.path.join(cur_dir, "*.jpg"), npz_path, chessboard_size, square_size, self.log)
            if res: messagebox.showinfo("完成", f"標定成功！RMS: {res['mean_reprojection_error']:.6f}")
        except Exception as e:
            self.log(f"\n[嚴重錯誤]: {e}"); messagebox.showerror("錯誤", str(e))
        finally:
            self.is_running = False; self.btn_start.config(state="normal", text="開始執行")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoCalibrateApp(root)
    root.mainloop()
