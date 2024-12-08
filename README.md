# Solar Energy Prediction

本專案使用 **LightGBM** 模型進行太陽能數據的訓練與預測，結合外部數據處理技術，實現高效的能源分析。

## 專案結構

- **final_train.csv**: 訓練數據集，包含多個特徵欄位。
- **requirements.txt**: 專案所需的套件清單。
- **LightGBM_solar.py**: 主程式，負責數據處理、模型訓練及評估。

## 環境設定

請確保已安裝 **Python 3.11.9** 的版本。

### 步驟 1: 安裝必要套件

執行以下指令安裝專案所需套件：

```bash
pip install -r requirements.txt
```

### 步驟 2: 準備數據

請將 final_train.csv 放置於專案根目錄

```bash
python LightGBM_solar.py
```

1. 加載訓練數據集 final_train.csv。
2. 進行數據清理及特徵工程。
3. 使用 LightGBM 訓練模型。
4. 輸出訓練結果。
