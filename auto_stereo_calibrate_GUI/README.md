# Auto Stereo Calibrate GUI (雙目相機標定系統)

雙目立體相機標定工具 (Stereo Calibration)。此系統設計用於在已知左右相機內參的情況下，計算雙目相機的外參 R 與基線 T。

## 安裝與需求

請確保已安裝以下 Python 套件：
```bash
pip install numpy opencv-python matplotlib scikit-learn
```
或是 pip install -r requirements.txt

## 使用說明

執行主程式：
```bash
python gui.py
```

### 操作流程

1.  **輸入影片**：
    - 選擇 **左視角影片** 與 **右視角影片**。

2.  **輸出設定**：
    - 選擇 **工作目錄**，系統將自動建立 `stereo/left` 與 `stereo/right` 資料夾結構。

3.  **輸入內參**：
    - 選擇左與右相機的單目標定結果檔 (`.npz`)。
    - 這些檔案應包含 `camera_matrix` (或 `mtx`) 與 `dist_coeffs` (或 `dist`)。

4.  **參數設定**：
    - **切幀間隔**：從影片提取圖片的頻率。
    - **棋盤格尺寸**：內角點數量 (例如 9x6)。
    - **方格大小**：棋盤格每個格子的實際邊長 (單位：公尺)。
    - **目標圖片組數**：最終挑選的成對圖片數量。

5.  **開始自動標定**：點擊按鈕，系統將自動執行以下步驟：
    - **Step 1**: 切割影片為圖片 (`origin_image`)。
    - **Step 2-3**: 進行雙目配對與智能挑選 (`final_image`)。
    - **Step 4**: 讀取輸入的內參。
    - **Step 5**: 執行 `cv2.stereoCalibrate` 計算 R, T。
    - **Step 6**: 輸出結果報告。

## 輸出檔案

- `stereo_rt.npz`: 最終標定結果。
    - `R`: 旋轉矩陣 (3x3)
    - `T`: 平移向量 (3x1)
    - `baseline`: 雙目基線長度 (m)
    - `ret`: 平均重投影誤差 (RMS)
- `calibration_selection_report.json`: 分析結果文字檔。
- `selection_analysis.png`: 圖片挑選的分佈分析圖表。
