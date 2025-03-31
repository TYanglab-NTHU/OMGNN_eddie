
# OMGNN: pKa 預測模型

## 專案概述

OMGNN (Organic Molecule Graph Neural Network) 是一個用於預測有機分子 pKa 值的深度學習系統。該系統基於圖形神經網路(GNN)，能夠從分子結構中學習特徵並準確預測解離常數的負對數(pKa)。本專案特別強化了對解離順序的預測和處理能力。

## 工作流程

整個 pKa 預測流程如下：

1. **資料收集與預處理**：獲取分子結構(SMILES)和對應的 pKa 值
2. **解離順序分析**：識別分子中的解離原子位置，並確定其解離順序
3. **特徵生成**：將分子轉換為圖形結構，編碼解離順序信息
4. **資料轉換**：轉換為 PyTorch Geometric 格式，保留順序信息
5. **模型訓練**：使用支持順序預測的圖形神經網路進行訓練
6. **模型評估**：評估分類、回歸和順序預測的準確性
7. **預測應用**：對新分子進行 pKa 值及其解離順序預測

## 核心程式檔案

### integrated_nist_converter.py

處理分子結構和解離順序識別：
- `predict_dissociable_atoms`: 識別可解離原子，現在支持順序信息
- `analyze_heterocyclic_rings`: 分析雜環結構
- `tensorize_molecule`: 將分子轉換為張量表示，包含解離順序信息

### pka_datautils.py

資料處理與格式轉換：
- `data_loader`: 處理帶有解離順序的資料
- `_create_dataset`: 創建包含解離順序的 PyTorch Geometric 資料集
- `evaluate_model`: 評估包括順序預測在內的模型性能

### pka_models.py

圖形神經網路模型架構：
- `SequentialBondMessagePassing`: 支持解離順序的訊息傳遞機制
- `PKA_GNN`: 增強的模型，現在可以：
  - 識別解離原子並預測 pKa 值
  - 預測原子的解離順序
  - 根據解離階段進行分層計算

### train_pka_model.py

訓練和評估流程：
- 支持解離順序資料的載入和處理
- 訓練過程中考慮解離順序的評估
- 按分子和解離階段聚合結果
- 全面評估包括順序預測在內的模型性能

## 資料格式

### 輸入格式

增強的 CSV 格式，現在包含以下列：
- `SMILES`: 分子的 SMILES 表示
- `pKa`: 實驗測量的 pKa 值
- `Dissociable_Atoms_Ordered`: 按解離順序排列的原子索引
- `Functional_Group_Ordered`: 對應的官能團類型

範例：
```
SMILES,pKa,Dissociable_Atoms_Ordered,Functional_Group_Ordered
CC(=O)O,4.76,2:0,Carboxylic_acid:Primary_amine
c1ccccc1O,9.95,6,Phenol
```

## 使用方法

### 環境設置

```bash
pip install rdkit torch torch-geometric pandas numpy scikit-learn
```

### 資料轉換（含解離順序）

```bash
python integrated_nist_converter.py --input your_nist_data.csv --output processed_data_with_order.csv --assign_order
```

### 模型訓練（支持解離順序）

```bash
python train_pka_model.py --input_data nist_pka_data_with_order.csv --version pka_seq_v1 --dissociable_atom_col Dissociable_Atoms_Ordered --func_group_col Functional_Group_Ordered --max_dissociation_steps 2
```

參數說明：
- `--input_data`: 含解離順序的數據路徑
- `--dissociable_atom_col`: 有序解離原子列名
- `--func_group_col`: 官能團類型列名
- `--max_dissociation_steps`: 最大解離階段數

## 特殊功能

### 解離順序預測

系統現在可以：
- 分析並確定分子中原子的解離順序
- 根據官能團類型和 pKa 值確定解離優先級
- 預測複雜分子中多個解離位點的順序
- 評估解離順序預測的準確性

### 分層模型結構

針對解離順序的分層設計：
- 每個解離階段使用專門的神經網絡層
- 考慮先前解離階段的影響
- 自適應處理不同解離階段數的分子
- 共享權重選項以減少參數數量

## 性能參考

擴展的評估指標：
- 分類準確率：>85% (解離原子識別)
- 回歸MSE：<1.0 (pKa值預測)
- MAE：<0.8 (平均絕對誤差)
- R²：>0.8 (決定係數)
- 解離順序準確率：>75% (新增指標)

## 開發者資訊

擴展系統支持新功能：
1. 在 `pka_models.py` 中擴展模型以支持更複雜的解離機制
2. 在 `pka_datautils.py` 中添加新的數據處理方法
3. 根據需要調整評估函數以測量新的性能指標

## 版本歷史

- v1.0: 基礎功能實現
- v1.1: 增強環狀化合物支持
- v1.2: 添加雜環結構處理
- v1.3: 整合日誌控制功能
- v2.0: 添加解離順序預測和處理支持
