import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path so utils can be imported if run from here
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import video_processor, auto_picker, calibrator

class AutoCalibrateApp:
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
        # Mode Section
        frame_mode = tk.LabelFrame(self.root, text="執行模式", font=self.label_font, padx=10, pady=10)
        frame_mode.pack(fill="x", padx=10, pady=5)
        
        self.combo_mode = ttk.Combobox(frame_mode, values=[
            self.MODE_VIDEO, self.MODE_ORIGIN, self.MODE_BEST, self.MODE_FINAL
        ], state="readonly", font=self.label_font, width=40)
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
        mode = self.combo_mode.get()
        self.entry_input.delete(0, tk.END)
        self.entry_input.config(state="normal")
        self.btn_browse_input.config(state="normal")
        
        if mode == self.MODE_VIDEO:
            self.lbl_input.config(text="影片路徑:")
            self.entry_interval.config(state="normal")
        elif mode == self.MODE_ORIGIN:
            self.lbl_input.config(text="原始圖片資料夾 (origin_image):")
            self.entry_interval.config(state="disabled")
        elif mode == self.MODE_BEST:
            self.lbl_input.config(text="角點篩選後圖片資料夾 (best_image):")
            self.entry_interval.config(state="disabled")
        elif mode == self.MODE_FINAL:
            self.lbl_input.config(text="自動選取後圖片資料夾 (final_img):")
            self.entry_interval.config(state="disabled")

    def browse_input(self):
        mode = self.combo_mode.get()
        if mode == self.MODE_VIDEO:
            filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
            if filename:
                self.entry_input.delete(0, tk.END)
                self.entry_input.insert(0, filename)
        else:
            foldername = filedialog.askdirectory()
            if foldername:
                self.entry_input.delete(0, tk.END)
                self.entry_input.insert(0, foldername)
            
    def browse_output(self):
        foldername = filedialog.askdirectory()
        if foldername:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, foldername)

    def browse_npz(self):
        filename = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if filename:
            self.entry_npz.delete(0, tk.END)
            self.entry_npz.insert(0, filename)
            
    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        print(message)
        
    def inspect_npz(self):
        filename = self.entry_npz.get()
        if not filename or not os.path.exists(filename):
            messagebox.showerror("錯誤", "請先選擇有效的 .npz 檔案")
            return
            
        try:
            data = np.load(filename, allow_pickle=True)
            info = []
            info.append(f"檔案: {os.path.basename(filename)}")
            info.append("-" * 30)
            
            for key in data.files:
                val = data[key]
                info.append(f"Key: {key}")
                if isinstance(val, np.ndarray):
                    info.append(f"  Shape: {val.shape}")
                    if val.size < 10:
                        info.append(f"  Value: {val}")
                elif isinstance(val, dict) or (val.dtype == 'O' and val.ndim == 0):
                    # Try to handle object/dict stored in npz
                    item = val.item() if val.shape == () else val
                    info.append(f"  Content: {item}")
                else:
                    info.append(f"  Type: {type(val)}")
                info.append("")
                
            # Show in a new window
            top = tk.Toplevel(self.root)
            top.title("NPZ 內容")
            top.geometry("600x500")
            text_area = scrolledtext.ScrolledText(top, font=('Consolas', 12))
            text_area.pack(fill="both", expand=True)
            text_area.insert(tk.END, "\n".join(info))
            
        except Exception as e:
            messagebox.showerror("錯誤", f"無法讀取 NPZ: {e}")

    def start_process(self):
        if self.is_running:
            return
            
        mode = self.combo_mode.get()
        input_path = self.entry_input.get()
        output_base = self.entry_output.get()
        
        if not output_base:
            messagebox.showerror("錯誤", "請選擇輸出目錄")
            return
            
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("錯誤", "輸入路徑無效")
            return
            
        try:
            interval = int(self.entry_interval.get())
            cols = int(self.entry_cols.get())
            rows = int(self.entry_rows.get())
            square_size = float(self.entry_square.get())
            target_count = int(self.entry_count.get())
        except ValueError:
            messagebox.showerror("錯誤", "參數格式錯誤，請檢查數字欄位")
            return
            
        self.is_running = True
        self.btn_start.config(state="disabled", text="執行中...")
        self.log_area.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.run_pipeline, 
                                  args=(mode, input_path, output_base, interval, (cols, rows), square_size, target_count))
        thread.start()
        
    def run_pipeline(self, mode, input_path, output_base, interval, chessboard_size, square_size, target_count):
        try:
            # Prepare default output folders
            origin_dir = os.path.join(output_base, "origin_image")
            best_dir = os.path.join(output_base, "best_image")
            final_dir = os.path.join(output_base, "final_img")
            
            # Step 1: Extract Frames
            if mode == self.MODE_VIDEO:
                self.log("Step 1/6: 開始提取影片幀...")
                # input_path is Video File
                count = video_processor.extract_frames(input_path, origin_dir, interval, progress_callback=self.log)
                self.log(f"-> 提取完成，共 {count} 張圖片儲存於 {origin_dir}")
                current_input_dir = origin_dir # Pass to next step
            else:
                self.log("Step 1: 跳過影片提取 (依據選擇模式)")
                current_input_dir = None
            
            # Step 2 & 3: Filter Valid Images
            if mode in [self.MODE_VIDEO, self.MODE_ORIGIN]:
                if mode == self.MODE_ORIGIN:
                    # input_path is Origin Folder
                    current_input_dir = input_path
                    # We might want to use the user-provided folder as 'origin_dir' for clarity using absolute paths?
                    # But the filter function copies FROM source TO best_dir.
                    # So current_input_dir is correct.
                
                self.log(f"\nStep 2 & 3: 篩選完整棋盤格圖片 (來源: {current_input_dir})...")
                
                if not os.path.exists(current_input_dir) or not os.listdir(current_input_dir):
                     raise Exception(f"原始圖片目錄不存在或為空: {current_input_dir}")
                     
                valid_count = video_processor.filter_valid_images(current_input_dir, best_dir, chessboard_size, progress_callback=self.log)
                self.log(f"-> 篩選完成，共 {valid_count} 張有效圖片儲存於 {best_dir}")
                
                if valid_count == 0:
                    raise Exception("沒有找到任何有效的棋盤格圖片，流程終止。")
            else:
                self.log("Step 2 & 3: 跳過圖片篩選 (依據選擇模式)")

            # Step 4: Auto Pick
            if mode in [self.MODE_VIDEO, self.MODE_ORIGIN, self.MODE_BEST]:
                if mode == self.MODE_BEST:
                    best_dir = input_path # Override implicit best_dir with user input
                
                self.log(f"\nStep 4: 智慧挑選最佳的 {target_count} 張圖片 (來源: {best_dir})...")
                if not os.path.exists(best_dir) or not os.listdir(best_dir):
                     raise Exception(f"篩選後圖片目錄不存在或為空: {best_dir}")
                     
                auto_picker.run_auto_pick(best_dir, final_dir, chessboard_size, square_size, target_count, logger=self.log)
                self.log(f"-> 挑選完成，結果儲存於 {final_dir}")
            else:
                 self.log("Step 4: 跳過智慧挑選 (依據選擇模式)")

            # Step 5 & 6: Calibration
            self.log("\nStep 5 & 6: 執行最終標定並輸出報告...")
            
            if mode == self.MODE_FINAL:
                final_dir = input_path # Override implicit final_dir with user input
                
            source_pattern = os.path.join(final_dir, "*.jpg")
            if not video_processor.Path(final_dir).exists():
                 raise Exception(f"最終圖片目錄不存在: {final_dir}")
            
            npz_path = os.path.join(output_base, "calibration_result.npz")
            
            results = calibrator.calibration_final(source_pattern, npz_path, chessboard_size, square_size, logger=self.log)
            
            if results:
                self.log(f"-> 標定成功！")
                self.log(f"平均重投影誤差: {results['mean_reprojection_error']:.6f}")
                messagebox.showinfo("完成", "全自動標定流程已成功完成！")
            else:
                self.log("-> 標定失敗，請檢查日誌。")
                messagebox.showwarning("失敗", "標定過程發生錯誤")

        except Exception as e:
            self.log(f"\n[嚴重錯誤] 流程發生異常: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("錯誤", f"發生異常: {str(e)}")
        finally:
            self.is_running = False
            self.btn_start.config(state="normal", text="開始執行")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoCalibrateApp(root)
    root.mainloop()
