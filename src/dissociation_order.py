import pandas as pd
import re
from rdkit import Chem
import os
import numpy as np

# 定義不同官能團的典型pKa範圍
PKA_RANGES = {
    'carboxylic_acid': (1.5, 5.0),      # 羧酸
    'phosphoric_acid': (1.0, 3.0),      # 磷酸第一解離
    'phosphoric_acid_2': (5.0, 7.0),    # 磷酸第二解離
    'phosphoric_acid_3': (9.0, 12.0),   # 磷酸第三解離
    'sulfonic_acid': (0.0, 2.0),        # 磺酸
    'phenol': (8.0, 11.0),              # 酚
    'thiol': (8.0, 11.0),               # 硫醇
    'amine_primary': (9.0, 11.0),       # 一級胺
    'amine_secondary': (9.5, 11.5),     # 二級胺
    'amine_tertiary': (9.0, 11.0),      # 三級胺
    'pyridine': (5.0, 6.5),             # 吡啶
    'imidazole': (6.5, 7.5),            # 咪唑
    'guanidine': (12.0, 14.0),          # 胍基
    'alcohol': (15.0, 18.0),            # 醇(不太可能在生理pH解離)
    'amide': (15.0, 17.0)               # 酰胺(不太可能在生理pH解離)
}

def determine_dissociation_order(smiles, dissociable_atoms_str, functional_groups_str, pka_value):
    """
    根據SMILES、可解離原子和官能團確定解離順序
    
    參數:
        smiles (str): 分子的SMILES表示
        dissociable_atoms_str (str): 可解離原子的索引，如 "0:2"
        functional_groups_str (str): 官能團類型，如 "amine_primary:carboxylic_acid"
        pka_value (float): 實驗測得的pKa值
        
    返回:
        tuple: (dissociable_atoms_ordered, functional_groups_ordered)
        解離順序的原子索引和對應功能團列表
    """
    # 解析可解離原子和官能團
    atom_indices = [int(idx) for idx in dissociable_atoms_str.split(':') if idx]
    functional_groups = functional_groups_str.split(':')
    
    # 檢查數量是否匹配
    if len(atom_indices) != len(functional_groups):
        print(f"警告: 原子索引數量({len(atom_indices)})與官能團數量({len(functional_groups)})不匹配: {smiles}")
        # 嘗試修復不匹配情況
        min_len = min(len(atom_indices), len(functional_groups))
        atom_indices = atom_indices[:min_len]
        functional_groups = functional_groups[:min_len]
    
    # 將原子索引與官能團配對
    atom_group_pairs = list(zip(atom_indices, functional_groups))
    
    # 根據官能團的典型pKa範圍排序
    def get_pka_range_midpoint(group):
        pka_range = PKA_RANGES.get(group, (14, 14))  # 預設值為高pKa
        return (pka_range[0] + pka_range[1]) / 2
    
    # 按照官能團的典型pKa中點值排序
    sorted_pairs = sorted(atom_group_pairs, key=lambda x: get_pka_range_midpoint(x[1]))
    
    # 提取排序後的原子索引和官能團
    sorted_atom_indices = [pair[0] for pair in sorted_pairs]
    sorted_functional_groups = [pair[1] for pair in sorted_pairs]
    
    # 考慮實際pKa與理論pKa的關係進行調整
    # 如果實際pKa值與某個官能團的pKa範圍更接近，則可能是該官能團在解離
    if pka_value is not None and not np.isnan(pka_value):
        # 計算實際pKa與各官能團pKa中點的距離
        distances = []
        for group in sorted_functional_groups:
            midpoint = get_pka_range_midpoint(group)
            distance = abs(pka_value - midpoint)
            distances.append(distance)
        
        # 找出最接近的官能團
        closest_idx = distances.index(min(distances))
        
        # 如果最接近的不是第一個，可能需要調整順序
        if closest_idx > 0 and distances[closest_idx] < 2.0:  # 如果距離小於2個pKa單位
            # 將該官能團及對應的原子索引提前
            group_to_move = sorted_functional_groups[closest_idx]
            atom_to_move = sorted_atom_indices[closest_idx]
            
            # 根據pKa值確定應放置的位置
            insert_position = 0
            for i, group in enumerate(sorted_functional_groups[:closest_idx]):
                if get_pka_range_midpoint(group) < pka_value:
                    insert_position = i + 1
            
            # 移除原位置
            sorted_functional_groups.pop(closest_idx)
            sorted_atom_indices.pop(closest_idx)
            
            # 插入新位置
            sorted_functional_groups.insert(insert_position, group_to_move)
            sorted_atom_indices.insert(insert_position, atom_to_move)
    
    # 將排序後的原子索引轉回原始格式
    dissociable_atoms_ordered = ':'.join(map(str, sorted_atom_indices))
    functional_groups_ordered = ':'.join(sorted_functional_groups)
    
    return dissociable_atoms_ordered, functional_groups_ordered

def assign_dissociation_order(input_file, output_file):
    """
    處理輸入文件，添加解離順序信息後保存到輸出文件
    """
    # 載入CSV文件
    df = pd.read_csv(input_file)
    
    # 檢查所需列是否存在
    required_columns = ['SMILES', 'pKa', 'Dissociable_Atoms', 'Functional_Group']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"輸入文件缺少必要的列: {col}")
    
    # 創建新列來存儲排序後的結果
    df['Dissociable_Atoms_Ordered'] = ""
    df['Functional_Group_Ordered'] = ""
    df['Dissociation_Order'] = ""  # 存儲解離順序說明
    
    # 處理每一行
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        pka = row['pKa']
        dissociable_atoms = row['Dissociable_Atoms']
        functional_groups = row['Functional_Group']
        
        # 確定解離順序
        try:
            atoms_ordered, groups_ordered = determine_dissociation_order(
                smiles, dissociable_atoms, functional_groups, pka)
            
            # 更新DataFrame
            df.at[idx, 'Dissociable_Atoms_Ordered'] = atoms_ordered
            df.at[idx, 'Functional_Group_Ordered'] = groups_ordered
            
            # 創建解離順序說明
            order_text = " > ".join([f"{a}({g})" for a, g in zip(
                atoms_ordered.split(':'), groups_ordered.split(':'))])
            df.at[idx, 'Dissociation_Order'] = order_text
            
        except Exception as e:
            print(f"處理行 {idx} 時出錯: {e}, SMILES: {smiles}")
            # 保持原始值
            df.at[idx, 'Dissociable_Atoms_Ordered'] = dissociable_atoms
            df.at[idx, 'Functional_Group_Ordered'] = functional_groups
            df.at[idx, 'Dissociation_Order'] = "處理錯誤"
    
    # 保存結果
    df.to_csv(output_file, index=False)
    print(f"處理完成: {len(df)} 行數據已保存至 {output_file}")
    
    # 返回處理後的統計信息
    ordered_count = df['Dissociation_Order'].ne("處理錯誤").sum()
    print(f"成功處理比例: {ordered_count}/{len(df)} ({ordered_count/len(df)*100:.2f}%)")
    
    return df

# 使用範例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='為分子數據添加解離順序標注')
    parser.add_argument('--input', default='../data/nist_pka_data_clean.csv',
                       help='輸入CSV文件路徑')
    parser.add_argument('--output', default='../data/nist_pka_data_with_order.csv',
                       help='輸出CSV文件路徑')
    
    args = parser.parse_args()
    
    result_df = assign_dissociation_order(args.input, args.output)
    
    # 顯示範例結果
    print("\n處理結果範例:")
    print(result_df[['SMILES', 'pKa', 'Dissociable_Atoms', 'Dissociable_Atoms_Ordered', 'Dissociation_Order']].head(10))
