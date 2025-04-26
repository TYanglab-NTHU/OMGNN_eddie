# pKa 預測模型使用指南

本文檔說明如何使用 pKa 預測模型進行單個分子的 pKa 位點預測以及批量處理多個分子。

## 功能概述

該模型使用圖神經網絡(GNN)來預測分子中的 pKa 位點及其對應的 pKa 值。模型具有以下功能：

1. **分類任務**：識別分子中具有 pKa 活性的原子位置
2. **迴歸任務**：預測這些位置的實際 pKa 值
3. **可視化**：生成帶有標注的分子結構圖，顯示 pKa 位點和數值

## 環境要求

- Python 3.7+
- PyTorch 1.7+
- PyTorch Geometric
- RDKit
- pandas
- numpy
- matplotlib
- tqdm (用於批量處理)

## 單個分子預測

使用 `pka_predict_sample.py` 腳本對單個分子進行 pKa 預測：

```bash
python src/pka_predict_sample.py --model path/to/model.pt --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

### 參數說明

- `--model`: 預訓練的模型路徑（必需）
- `--smiles`: 要預測的分子SMILES字符串（必需）
- `--device`: 使用設備，可選 'cuda' 或 'cpu'（默認：自動選擇）
- `--output`: 可視化圖像保存路徑（可選）
- `--no-viz`: 添加此參數禁用可視化（可選）

### 示例

```bash
# 預測阿司匹林的pKa值並保存可視化結果
python src/pka_predict_sample.py --model models/pka_model_v1.pt --smiles "CC(=O)Oc1ccccc1C(=O)O" --output results/aspirin_pka.png

# 預測苯酚的pKa值但不生成可視化
python src/pka_predict_sample.py --model models/pka_model_v1.pt --smiles "c1ccc(cc1)O" --no-viz
```

## 批量預測

使用 `pka_batch_predict.py` 腳本對多個分子進行批量 pKa 預測：

```bash
python src/pka_batch_predict.py --model path/to/model.pt --input molecules.csv --output predictions.csv
```

### 參數說明

- `--model`: 預訓練的模型路徑（必需）
- `--input`: 包含SMILES的CSV文件路徑（必需）
- `--smiles-col`: SMILES列的名稱（默認：'smiles'）
- `--output`: 保存結果的CSV文件路徑（必需）
- `--viz-dir`: 保存可視化結果的目錄（可選）
- `--device`: 使用設備，可選 'cuda' 或 'cpu'（默認：自動選擇）
- `--no-parallel`: 添加此參數禁用並行處理（可選）
- `--workers`: 並行處理的工作者數量（可選，默認：CPU核心數）

### 輸入CSV格式

輸入CSV文件至少應包含一列SMILES字符串，例如：

```csv
smiles
CC(=O)Oc1ccccc1C(=O)O
c1ccc(cc1)O
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
```

### 輸出CSV格式

輸出CSV文件包含以下列：

- `smiles`: 原始SMILES字符串
- `success`: 預測是否成功（True/False）
- `num_pka_sites`: 發現的pKa位點數量
- `pka_positions`: 逗號分隔的pKa位點索引
- `pka_values`: 逗號分隔的pKa值

### 示例

```bash
# 批量預測分子庫的pKa值並保存可視化結果
python src/pka_batch_predict.py --model models/pka_model_v1.pt --input data/molecules.csv --output results/predictions.csv --viz-dir results/images

# 僅批量預測，不生成可視化
python src/pka_batch_predict.py --model models/pka_model_v1.pt --input data/molecules.csv --output results/predictions.csv --no-parallel
```

## 在自定義代碼中使用

您也可以在自己的Python代碼中直接使用該模型：

```python
import torch
from self_pka_models import pka_GNN

# 載入模型
node_dim = 153    # 節點特徵維度
bond_dim = 11     # 鍵特徵維度
hidden_dim = 128  # 隱藏層維度
output_dim = 1    # 輸出維度
dropout = 0.2     # Dropout率

# 初始化模型
model = pka_GNN(node_dim, bond_dim, hidden_dim, output_dim, dropout)

# 載入預訓練權重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('models/pka_model_v1.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 預測單個分子
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 阿司匹林
result = model.sample(smiles, device)

# 處理結果
if result:
    print(f"SMILES: {smiles}")
    print("預測的pKa位置及數值:")
    for i, pos in enumerate(result['pka_positions']):
        print(f"  原子 {pos}: pKa = {result['pka_values'][i]:.2f}")
```

## 常見問題

### 為什麼我的分子沒有預測到pKa位點？

並非所有分子都具有pKa活性位點。模型預測的是原子是否具有pKa活性，如果沒有位點被預測為活性，可能有以下原因：

1. 分子確實沒有酸鹼活性部分
2. 模型未能識別出該類型的pKa活性位點
3. 模型在該類分子上的泛化能力有限

### 如何解讀pKa位點？

pKa位點的索引對應於RDKit分子對象中的原子索引。您可以使用RDKit工具查看分子中的原子索引，或使用提供的可視化功能直觀地查看結果。

### 模型預測的pKa值準確嗎？

pKa值的預測準確性取決於模型訓練數據的質量和覆蓋範圍。對於常見的官能團（如羧酸、胺、酚等），模型通常能提供合理的預測；但對於不常見的結構，預測結果可能不夠準確。

## 引用

如果您在研究中使用此模型，請引用：

```
待添加論文引用信息
``` 