import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path so utils can be imported if run from here
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import video_processor, stereo_picker, calibrator

class AutoStereoCalibrateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("自動雙目標定系統")
        self.root.geometry("700x900")
        
        self.is_running = False
        
        # Styles
        self.label_font = ('Microsoft JhengHei', 10)
        self.entry_font = ('Arial', 10)
        
        # Modes
        self.MODE_FULL = "從影片開始 (完整流程)"
        # self.MOOE_SKIP_VIDEO = "從原始圖片 (origin_image) 開始" # 可擴充
        
        self.create_widgets()
        
    def create_widgets(self):
        # 1. Input Videos Section
        frame_videos = tk.LabelFrame(self.root, text="1. 輸入影片 (Left / Right)", font=self.label_font, padx=10, pady=10)
        frame_videos.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_videos, text="左視角影片:", font=self.label_font).grid(row=0, column=0, sticky="w")
        self.entry_video_L = tk.Entry(frame_videos, width=50, font=self.entry_font)
        self.entry_video_L.grid(row=0, column=1, padx=5)
        tk.Button(frame_videos, text="瀏覽", command=lambda: self.browse_file(self.entry_video_L, "Video"), font=self.label_font).grid(row=0, column=2)

        tk.Label(frame_videos, text="右視角影片:", font=self.label_font).grid(row=1, column=0, sticky="w")
        self.entry_video_R = tk.Entry(frame_videos, width=50, font=self.entry_font)
        self.entry_video_R.grid(row=1, column=1, padx=5)
        tk.Button(frame_videos, text="瀏覽", command=lambda: self.browse_file(self.entry_video_R, "Video"), font=self.label_font).grid(row=1, column=2)

        # 2. Output Directory
        frame_out = tk.LabelFrame(self.root, text="2. 輸出設定", font=self.label_font, padx=10, pady=10)
        frame_out.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_out, text="工作目錄 (Stereo Folder):", font=self.label_font).grid(row=0, column=0, sticky="w")
        self.entry_output = tk.Entry(frame_out, width=50, font=self.entry_font)
        self.entry_output.grid(row=0, column=1, padx=5)
        tk.Button(frame_out, text="瀏覽", command=self.browse_output, font=self.label_font).grid(row=0, column=2)

        # 3. Intrinsics Section
        frame_intrinsics = tk.LabelFrame(self.root, text="3. 輸入內參 (Left / Right .npz)", font=self.label_font, padx=10, pady=10)
        frame_intrinsics.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_intrinsics, text="左相機 NPZ:", font=self.label_font).grid(row=0, column=0, sticky="w")
        self.entry_npz_L = tk.Entry(frame_intrinsics, width=50, font=self.entry_font)
        self.entry_npz_L.grid(row=0, column=1, padx=5)
        tk.Button(frame_intrinsics, text="瀏覽", command=lambda: self.browse_file(self.entry_npz_L, "NPZ"), font=self.label_font).grid(row=0, column=2)
        
        tk.Label(frame_intrinsics, text="右相機 NPZ:", font=self.label_font).grid(row=1, column=0, sticky="w")
        self.entry_npz_R = tk.Entry(frame_intrinsics, width=50, font=self.entry_font)
        self.entry_npz_R.grid(row=1, column=1, padx=5)
        tk.Button(frame_intrinsics, text="瀏覽", command=lambda: self.browse_file(self.entry_npz_R, "NPZ"), font=self.label_font).grid(row=1, column=2)

        # 4. Parameters Section
        frame_params = tk.LabelFrame(self.root, text="4. 參數設定", font=self.label_font, padx=10, pady=10)
        frame_params.pack(fill="x", padx=10, pady=5)
        
        # Row 0
        tk.Label(frame_params, text="切幀間隔:", font=self.label_font).grid(row=0, column=0, sticky="w")
        self.entry_interval = tk.Entry(frame_params, width=10, font=self.entry_font)
        self.entry_interval.insert(0, "5")
        self.entry_interval.grid(row=0, column=1, sticky="w")
        
        tk.Label(frame_params, text="棋盤格尺寸 (Cols x Rows):", font=self.label_font).grid(row=0, column=2, sticky="w", padx=(10,0))
        frame_size = tk.Frame(frame_params)
        frame_size.grid(row=0, column=3, sticky="w")
        self.entry_cols = tk.Entry(frame_size, width=5, font=self.entry_font)
        self.entry_cols.insert(0, "9")
        self.entry_cols.pack(side="left")
        tk.Label(frame_size, text="x").pack(side="left")
        self.entry_rows = tk.Entry(frame_size, width=5, font=self.entry_font)
        self.entry_rows.insert(0, "6")
        self.entry_rows.pack(side="left")
        
        # Row 1
        tk.Label(frame_params, text="方格大小 (m):", font=self.label_font).grid(row=1, column=0, sticky="w")
        self.entry_square = tk.Entry(frame_params, width=10, font=self.entry_font)
        self.entry_square.insert(0, "0.09")
        self.entry_square.grid(row=1, column=1, sticky="w")

        tk.Label(frame_params, text="目標圖片組數:", font=self.label_font).grid(row=1, column=2, sticky="w", padx=(10,0))
        self.entry_count = tk.Entry(frame_params, width=10, font=self.entry_font)
        self.entry_count.insert(0, "15")
        self.entry_count.grid(row=1, column=3, sticky="w")

        # Action Section
        self.btn_start = tk.Button(self.root, text="開始自動標定", command=self.start_process, 
                                   bg="#4CAF50", fg="white", font=('Microsoft JhengHei', 14, 'bold'), height=2)
        self.btn_start.pack(fill="x", padx=20, pady=15)
        
        # Log Section
        tk.Label(self.root, text="執行紀錄:", font=self.label_font).pack(anchor="w", padx=10)
        self.log_area = scrolledtext.ScrolledText(self.root, height=15, font=('Consolas', 9))
        self.log_area.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def browse_file(self, entry_widget, file_type):
        if file_type == "Video":
            ftypes = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        elif file_type == "NPZ":
            ftypes = [("NPZ files", "*.npz")]
        else:
            ftypes = []
            
        filename = filedialog.askopenfilename(filetypes=ftypes)
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)

    def browse_output(self):
        foldername = filedialog.askdirectory()
        if foldername:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, foldername)

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        print(message)

    def load_intrinsics(self, npz_path):
        try:
            data = np.load(npz_path)
            # 嘗試讀取常見的 key names
            if 'camera_matrix' in data: mtx = data['camera_matrix']
            elif 'mtx' in data: mtx = data['mtx']
            else: return None, None, f"找不到相機矩陣 (keys: {list(data.keys())})"
            
            if 'dist_coeffs' in data: dist = data['dist_coeffs']
            elif 'dist' in data: dist = data['dist']
            else: return None, None, f"找不到畸變係數 (keys: {list(data.keys())})"
            
            return mtx, dist, None
        except Exception as e:
            return None, None, str(e)

    def start_process(self):
        if self.is_running: return
        
        # 1. Gather Inputs
        video_L = self.entry_video_L.get()
        video_R = self.entry_video_R.get()
        output_base = self.entry_output.get()
        npz_L = self.entry_npz_L.get()
        npz_R = self.entry_npz_R.get()
        
        # Validation
        if not all([video_L, video_R, output_base, npz_L, npz_R]):
            messagebox.showerror("錯誤", "所有欄位都必須填寫！")
            return
            
        if not all(map(os.path.exists, [video_L, video_R, npz_L, npz_R])):
             messagebox.showerror("錯誤", "部分輸入檔案不存在，請檢查路徑。")
             return

        try:
            interval = int(self.entry_interval.get())
            cols = int(self.entry_cols.get())
            rows = int(self.entry_rows.get())
            square_size = float(self.entry_square.get())
            target_count = int(self.entry_count.get())
            chessboard_size = (cols, rows)
        except ValueError:
            messagebox.showerror("錯誤", "數字參數格式錯誤")
            return

        self.is_running = True
        self.btn_start.config(state="disabled", text="執行中...")
        self.log_area.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.run_pipeline, 
                                  args=(video_L, video_R, output_base, npz_L, npz_R, interval, chessboard_size, square_size, target_count))
        thread.start()

    def run_pipeline(self, video_L, video_R, output_base, npz_L_path, npz_R_path, interval, chessboard_size, square_size, target_count):
        try:
            # Prepare Directories
            stereo_dir = os.path.join(output_base, "stereo")
            left_origin = os.path.join(stereo_dir, "left", "origin_image")
            right_origin = os.path.join(stereo_dir, "right", "origin_image")
            left_final = os.path.join(stereo_dir, "left", "final_image")
            right_final = os.path.join(stereo_dir, "right", "final_image")
            
            # 0. Load Intrinsics
            self.log("Step 0: 讀取內參...")
            mtxL, distL, errL = self.load_intrinsics(npz_L_path)
            if errL: raise Exception(f"左相機 NPZ 錯誤: {errL}")
            
            mtxR, distR, errR = self.load_intrinsics(npz_R_path)
            if errR: raise Exception(f"右相機 NPZ 錯誤: {errR}")
            
            self.log("內參讀取成功。")

            # 1. Video Frame Extraction
            self.log("\nStep 1/6: 影片切幀...")
            self.log(f"處理左影片: {os.path.basename(video_L)}")
            cL = video_processor.extract_frames(video_L, left_origin, interval, self.log)
            self.log(f"-> 左影片提取 {cL} 張")
            
            self.log(f"處理右影片: {os.path.basename(video_R)}")
            cR = video_processor.extract_frames(video_R, right_origin, interval, self.log)
            self.log(f"-> 右影片提取 {cR} 張")

            # 2. Filter (Implicit in Picker, but we ensure folders exist)
            
            # 3. Stereo Picking
            self.log(f"\nStep 3: 執行 @auto_pick_stereo_img (目標: {target_count} 組)...")
            count, final_rms = stereo_picker.run_stereo_pick(
                left_origin, right_origin, 
                left_final, right_final,
                mtxL, distL, mtxR, distR,
                chessboard_size, target_count,
                logger=self.log
            )
            
            if count == 0:
                raise Exception("自動挑選失敗，無有效配對圖片。")
            
            self.log(f"-> 挑選完成，共 {count} 組，RMS: {final_rms:.4f}")

            # 4. Load NPZs Input (Already done in step 0, just ensuring flow)
            
            # 5 & 6. Stereo Calibration
            self.log(f"\nStep 5: 進行雙目標定並輸出結果...")
            stereo_npz_path = os.path.join(output_base, "stereo_rt.npz")
            
            res = calibrator.stereo_calibration(
                left_final, right_final,
                mtxL, distL, mtxR, distR,
                stereo_npz_path,
                chessboard_size, square_size,
                logger=self.log
            )
            
            if res:
                self.log("\n====== 標定結果 ======")
                self.log(f"Ret (RMS): {res['ret']:.6f}")
                self.log(f"Baseline: {res['baseline']:.6f} m")
                self.log(f"Translation Vector (T):\n{res['T'].flatten()}")
                # self.log(f"Rotation Matrix (R):\n{res['R']}")
                self.log(f"\n結果已存於: {stereo_npz_path}")
                messagebox.showinfo("完成", "雙目標定流程已成功完成！")
            else:
                self.log("標定失敗。")
                messagebox.showwarning("失敗", "標定過程發生錯誤")

        except Exception as e:
            self.log(f"\n[嚴重錯誤] {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("錯誤", f"發生異常: {str(e)}")
        finally:
            self.is_running = False
            self.btn_start.config(state="normal", text="開始自動標定")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoStereoCalibrateApp(root)
    root.mainloop()
