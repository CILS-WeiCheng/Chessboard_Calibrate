# Auto Calibrate GUI (單目相機標定系統)

影片逐幀切圖片、有效角點的棋盤格篩選、自動圖片挑選以及相機內參、畸變係數標定的自動化流程。

## 安裝與需求

請確保已安裝以下 Python 套件：
```bash
pip install numpy opencv-python matplotlib scikit-learn
```
或是 ```bash
pip install -r requirements.txt
```
## 使用說明

執行主程式：
```bash
python gui.py
```

### 操作流程

1.  **選擇執行模式**：
    - **從影片開始**：輸入影片檔，程式會自動切幀 -> 篩選 -> 挑選 -> 標定。
    - **從原始圖片 (origin_image)**：輸入影片圖幀切過後的資料夾，程式會篩選 -> 挑選 -> 標定。
    - **從篩選後圖片 (best_image)**：輸入已經過初步角點篩選的資料夾，程式會挑選 -> 標定。
    - **從最終圖片 (final_img)**：直接對角點完整的的圖片進行自動選取。

2.  **路徑設定**：
    - **影片/圖片路徑**：依據模式選擇對應的輸入檔案或資料夾。
    - **輸出/工作目錄**：結果檔案將儲存於此目錄下的 `origin_image`, `best_image`, `final_img` 資料夾中。

3.  **參數設定**：
    - **切幀間隔**：每幾幀存一張圖 (僅影片模式有效)。
    - **棋盤格尺寸**：內角點數量 (例如 9x6)。
    - **方格大小**：棋盤格每個格子的實際邊長 (單位：公尺)。
    - **目標圖片數量**：最終希望挑選幾張圖片進行標定 (建議 15-20 張)。

4.  **開始執行**：點擊按鈕開始自動化流程。

## 輸出檔案

- `calibration_result.npz`: 包含標定結果 (camera_matrix, dist_coeffs, rvecs, tvecs 等)。
- `calibration_result.txt`: 標定結果文字檔。
- `selection_analysis.png`: 圖片挑選的分佈分析圖表。
