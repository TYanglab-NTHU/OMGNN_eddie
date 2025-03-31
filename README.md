
# OMGNN: pKa 預測模型

## 專案概述

OMGNN (Organic Molecule Graph Neural Network) 是一個用於預測有機分子 pKa 值的深度學習系統。該系統基於圖形神經網路(GNN)，能夠從分子結構中學習特徵並準確預測解離常數的負對數(pKa)。本專案特別強化了對環狀化合物、雜環化合物和特殊官能團的解析和處理能力。

## 工作流程

整個 pKa 預測流程如下：

1. **資料收集與預處理**：從 NIST 或其他來源獲取分子結構(SMILES)和對應的 pKa 值
2. **分子解析**：解析 SMILES 字串，識別分子中的解離原子位置
3. **特徵生成**：將分子轉換為圖形結構，包括原子特徵、鍵特徵等
4. **資料轉換**：轉換為 PyTorch Geometric 可用的資料格式
5. **模型訓練**：使用圖形神經網路訓練 pKa 預測模型
6. **模型評估**：在測試集上評估模型性能
7. **預測應用**：對新分子進行 pKa 預測

## 核心程式檔案

### integrated_nist_converter.py

此檔案專門處理 NIST 資料庫中的分子結構，特別針對複雜環狀化合物進行優化。

主要功能：
- `sanitize_smiles`: 修復有問題的 SMILES 字串
- `predict_dissociable_atoms`: 識別分子中可能解離的原子位置
- `analyze_heterocyclic_rings`: 分析雜環結構
- `analyze_pyridine_structures`: 分析吡啶及其衍生物
- `analyze_imidazole_structures`: 分析咪唑及其衍生物
- `tensorize_molecule`: 將分子轉換為張量表示
- `convert_nist_data`: 將 NIST 資料轉換為模型可用格式

### pka_datautils.py

此檔案負責資料處理和 PyTorch Geometric 格式轉換。

主要功能：
- `load_config_by_version`: 根據版本載入配置參數
- `data_loader`: 載入和處理 pKa 資料
- `_create_dataset`: 創建 PyTorch Geometric 資料集
- `evaluate_model`: 評估模型性能

### pka_models.py

此檔案實現了 pKa 預測的圖形神經網路模型架構。

主要功能：
- `BondMessagePassing`: 實現分子中原子間的訊息傳遞機制
- `PKA_GNN`: 核心圖神經網路模型，包含：
  - 分類任務：識別分子中的可解離原子
  - 回歸任務：準確預測pKa值
  - 門控機制：使用RNN門控提升預測準確性
  - 前向傳播和預測功能

### train_pka_model.py

此檔案實現了模型的訓練和評估流程。

主要功能：
- 載入配置參數和訓練數據
- 初始化模型、優化器和學習率調度器
- 實現完整的訓練循環，包含：
  - 批次訓練和梯度更新
  - 定期評估模型性能
  - 記錄訓練歷史和指標
  - 模型保存和檢查點
- 提供最終模型評估和結果輸出
- 支援多種評估指標：分類準確率、MSE、MAE和R²

## 資料格式

### 輸入格式

輸入資料通常是 CSV 格式，包含以下列：
- `SMILES`: 分子的 SMILES 表示
- `pKa`: 實驗測量的 pKa 值
- `LigandName`(可選): 分子名稱
- `LigandClass`(可選): 分子類別
- `DissociableAtom`(可選): 可解離原子的索引，格式為逗號或冒號分隔的數字

範例：
```
SMILES,pKa,LigandName,LigandClass
CCO,15.9,Ethanol,Alcohol
CC(=O)O,4.76,Acetic acid,Carboxylic acid
c1ccccc1O,9.95,Phenol,Phenol
```

### 中間轉換格式

在處理過程中，分子被轉換為圖形表示：
- 節點：原子及其特徵
- 邊：化學鍵及其特徵
- 節點標籤：解離原子掩碼和 pKa 值

### 輸出格式

評估結果通常包含：
- 分類準確率 (解離原子識別)
- 回歸MSE (pKa預測誤差)
- 詳細結果 CSV，包含 SMILES、真實 pKa 和預測 pKa

## 使用方法

### 環境設置

```bash
pip install rdkit torch torch-geometric pandas numpy scikit-learn
```

### 資料轉換

```bash
python integrated_nist_converter.py --input your_nist_data.csv --output processed_data.csv --verbose
```

### 模型訓練與評估

```python
from pka_datautils import PKADataProcessor

# 載入資料
train_loader, test_loader = PKADataProcessor.data_loader(
    'processed_data.csv', 
    smiles_col='SMILES', 
    pka_col='pKa', 
    batch_size=32
)

# 訓練模型 (假設代碼)
from pka_model import PKAPredictor
model = PKAPredictor()
# ... 訓練代碼 ...

# 評估模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cls_acc, reg_mse, results = PKADataProcessor.evaluate_model(
    model, test_loader, device, 'evaluation_results.csv'
)
print(f"分類準確率: {cls_acc:.4f}, 回歸MSE: {reg_mse:.4f}")
```

### 完整訓練流程

使用訓練腳本可以快速開始訓練：

```bash
python train_pka_model.py --input_data your_data.csv --version your_model_version
```

腳本支援多種命令行參數：
- `--config_csv`: 配置文件路徑
- `--version`: 模型版本標識
- `--input_data`: 輸入數據路徑
- `--smiles_col`: SMILES列名
- `--pka_col`: pKa列名
- `--dissociable_atom_col`: 解離原子列名(可選)

## 特殊功能

### 日誌控制

系統默認不會顯示詳細日誌，但可以在需要調試時啟用：

```python
# 在 integrated_nist_converter 中
from integrated_nist_converter import set_verbose
set_verbose(True)  # 啟用詳細日誌

# 在 pka_datautils 中
from pka_datautils import PKADataProcessor
PKADataProcessor.set_verbose(True)  # 啟用詳細日誌
```

### 分子類型支持

本系統特別強化了對以下分子類型的支持：
- 環狀化合物
- 雜環化合物 (包括吡啶、咪唑、噻唑等)
- 經典有機官能團 (酸、醇、酚、胺)
- 特殊元素化合物 (含硫、氟、硒、砷等)

### 特殊結構映射

對於常見的特殊結構，系統內建了映射表，可以直接識別其解離原子位置：
- 氨基酸 (甘氨酸、丙氨酸、天冬氨酸等)
- 雜環結構 (吡啶、咪唑、菲羅啉等)
- 雙吡啶化合物
- 喹啉和噻唑衍生物

## 性能參考

在 NIST 資料庫的測試中：
- 原始處理成功率約 17% (1155/6738)
- 強化後處理成功率顯著提高，特別是對複雜環狀化合物

### 模型性能

典型的模型性能指標：
- 分類準確率：>85% (解離原子識別) 
- 回歸MSE：<1.0 (pKa值預測)
- MAE：<0.8 (平均絕對誤差)
- R²：>0.8 (決定係數)

## 進階配置

配置文件支持多種模型版本設置，可以通過版本號載入不同的配置：

```python
config = PKADataProcessor.load_config_by_version('config.csv', 'v1.0')
```

常用配置參數包括：
- `hidden_dim`: 隱藏層維度
- `batch_size`: 批次大小
- `num_epochs`: 訓練周期數
- `dropout`: 隨機失活率
- `lr`: 學習率
- `anneal_rate`: 學習率衰減因子

## 疑難排解

如遇到以下問題：

1. **SMILES 解析失敗**：
   - 嘗試使用 `sanitize_smiles` 函數修復
   - 檢查 SMILES 語法是否正確

2. **未找到解離原子**：
   - 考慮添加到 `SPECIAL_STRUCTURES_MAP`
   - 激活詳細日誌查看問題

3. **DataFrame 長度不一致錯誤**：
   - 已通過最新版本修復，確保使用最新代碼

## 開發者資訊

要擴展系統支持新的分子類型：

1. 在 `integrated_nist_converter.py` 中添加新的分析函數
2. 對於已知結構，擴展 `SPECIAL_STRUCTURES_MAP`
3. 在 `predict_dissociable_atoms` 函數中整合新的分析方法

## 引用與參考

本專案基於以下技術：
- RDKit: 分子操作工具庫
- PyTorch Geometric: 圖形神經網絡框架
- NIST 資料: 實驗pKa值來源

## 版本歷史

- v1.0: 基礎功能實現
- v1.1: 增強環狀化合物支持
- v1.2: 添加各種雜環結構的特殊處理
- v1.3: 整合日誌控制功能

## 貢獻指南

歡迎貢獻新的分子類型支持或性能優化。請確保：
1. 新增的代碼包含詳細註釋
2. 提供測試用例
3. 更新文檔以反映新功能
