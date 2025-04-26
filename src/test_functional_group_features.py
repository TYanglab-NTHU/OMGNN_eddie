import pandas as pd
import ast
import torch
import matplotlib.pyplot as plt
import numpy as np
from self_pka_chemutils import tensorize_for_pka, functional_group_features, atom_features

def test_functional_group_features():
    # 從CSV檔案讀取數據
    csv_path = '/work/u5066474/NTHU/LiveTransForM-main/OMGNN/output/pka_mapping_results.csv'
    df = pd.read_csv(csv_path)
    
    print(f"從CSV檔案讀取了 {len(df)} 筆數據")
    
    # 選擇一個範例分子進行測試
    example_idx = 0  # 第一個分子作為範例
    example_row = df.iloc[example_idx]
    
    smiles = example_row['smiles']
    functional_groups = ast.literal_eval(example_row['functional_groups'])
    pka_values = ast.literal_eval(example_row['pka_values'])
    mapping = ast.literal_eval(example_row['mapping'])
    
    print(f"SMILES: {smiles}")
    print(f"官能基: {functional_groups}")
    print(f"pKa值: {pka_values}")
    print(f"映射: {mapping}")
    
    # 使用我們的函數處理分子
    print("\n使用自定義functional_group_features處理分子:")
    
    # 1. 不使用官能基信息處理分子
    x_without_fg, edge_index_without_fg, edge_attr_without_fg = tensorize_for_pka(smiles)
    print(f"不使用官能基信息時的原子特徵形狀: {x_without_fg.shape}")
    
    # 2. 使用官能基信息處理分子
    x_with_fg, edge_index_with_fg, edge_attr_with_fg = tensorize_for_pka(smiles, functional_groups)
    print(f"使用官能基信息時的原子特徵形狀: {x_with_fg.shape}")
    
    # 檢查官能基特徵向量和原子特徵向量的差異
    for i, fg in enumerate(functional_groups):
        fg_type = fg['type']
        for idx in fg['indices']:
            print(f"\n官能基類型 {fg_type} (索引 {idx}):")
            print(f"  原子特徵向量維度: {x_without_fg[idx].shape}")
            print(f"  官能基特徵向量維度: {x_with_fg[idx].shape}")
            
            # 可視化特徵差異 (可選)
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.title(f"Atom Feature Vector (Index {idx})")
            plt.bar(range(len(x_without_fg[idx])), x_without_fg[idx].numpy())
            
            plt.subplot(2, 1, 2)
            plt.title(f"Functional Group Feature Vector (Index {idx}, Type {fg_type})")
            plt.bar(range(len(x_with_fg[idx])), x_with_fg[idx].numpy())
            
            plt.tight_layout()
            plt.savefig(f"feature_comparison_{fg_type}_{idx}.png")
            plt.close()
            
            break  # 只檢查每個官能基的第一個原子，避免輸出過多
    
    # 比較每種官能基類型的特徵
    print("\n比較不同官能基類型的特徵:")
    unique_fg_types = set(fg['type'] for fg in functional_groups)
    
    # 為每種官能基類型選擇一個代表性原子，生成特徵
    fg_type_to_feature = {}
    for fg in functional_groups:
        fg_type = fg['type']
        if fg_type not in fg_type_to_feature:
            # 取該官能基的第一個原子作為代表
            idx = fg['indices'][0]
            fg_type_to_feature[fg_type] = x_with_fg[idx].numpy()
    
    # 可視化不同官能基類型的特徵差異 (可選)
    if len(fg_type_to_feature) > 1:
        plt.figure(figsize=(12, 8))
        for i, (fg_type, feature) in enumerate(fg_type_to_feature.items()):
            plt.subplot(len(fg_type_to_feature), 1, i+1)
            plt.title(f"{fg_type} Feature Vector")
            plt.bar(range(len(feature)), feature)
        
        plt.tight_layout()
        plt.savefig("different_functional_groups_comparison.png")
        plt.close()
    
    print("測試完成！")

if __name__ == "__main__":
    test_functional_group_features() 