import pandas as pd
import os
from rdkit import Chem
from tqdm import tqdm
import sys
from self_pka_chemutils import predict_dissociable_atoms, get_dissociable_atoms_for_aromatics, fallback_smiles_analysis

def convert_nist_data(input_file, output_file):
    """全面增強的NIST數據轉換函數，專門針對環狀結構優化"""
    print(f"正在轉換 {input_file} 至 {output_file}...")
    
    try:
        # 讀取NIST數據
        df = pd.read_csv(input_file)
        print(f"載入了 {len(df)} 條數據記錄")
    except Exception as e:
        print(f"讀取輸入文件時出錯: {e}")
        return None
    
    # 創建結果數據框
    result_df = pd.DataFrame(columns=['SMILES', 'pKa', 'Dissociable_Atoms', 'Functional_Group', 'Name'])
    
    # 記錄處理統計
    processed = 0
    success = 0
    failed = 0
    error_log = []
    
    # 遍歷每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="處理分子"):
        try:
            smiles = row['SMILES']
            pka = row['Value'] if 'Value' in row else row['pKa'] if 'pKa' in row else None
            name = row['Ligand'] if 'Ligand' in row else row['Name'] if 'Name' in row else f"Compound-{idx}"
            
            if pd.isna(smiles) or not isinstance(smiles, str) or len(smiles) == 0:
                error_log.append(f"跳過索引 {idx}: 無效的SMILES")
                continue
                
            if pd.isna(pka) or pka is None:
                error_log.append(f"跳過索引 {idx}: 無效的pKa值")
                continue
            
            # 查看是否有max_eq_num欄位
            expected_num = None
            if 'max_eq_num' in row and pd.notna(row['max_eq_num']):
                try:
                    expected_num = int(row['max_eq_num'])
                except:
                    pass
            
            # 首先檢查是否是特殊SMILES
            if smiles in SPECIAL_STRUCTURES_MAP:
                dissociable_atoms = SPECIAL_STRUCTURES_MAP[smiles]
            else:
                # 使用增強版預測函數
                dissociable_atoms = predict_dissociable_atoms(smiles)
            
            processed += 1
            
            # 檢查與預期解離位點數
            if expected_num is not None and len(dissociable_atoms) != expected_num:
                print(f"警告: {smiles} 預期 {expected_num} 個解離位點，但找到 {len(dissociable_atoms)} 個")
                
                # 多層級嘗試找出預期數量的解離位點
                for attempt_func in [
                    lambda s: analyze_heterocyclic_rings(Chem.MolFromSmiles(s)),
                    lambda s: get_dissociable_atoms_for_aromatics(Chem.MolFromSmiles(s), s),
                    lambda s: comprehensive_atom_scan(Chem.MolFromSmiles(s)),
                    lambda s: analyze_smiles_directly(s)
                ]:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            continue
                            
                        alt_atoms = attempt_func(smiles if 's' in attempt_func.__code__.co_varnames else mol)
                        
                        if len(alt_atoms) == expected_num:
                            dissociable_atoms = alt_atoms
                            break
                        elif len(alt_atoms) > expected_num:
                            # 根據優先級選擇
                            priority_order = {
                                'carboxylic_acid': 0,
                                'amine_primary': 1,
                                'amine_secondary': 2,
                                'phenol': 3,
                                'alcohol': 4,
                                'thiol': 5,
                                'phosphate': 6,
                                'unknown': 999
                            }
                            
                            alt_atoms.sort(key=lambda x: priority_order.get(x[1], 999))
                            dissociable_atoms = alt_atoms[:expected_num]
                            break
                    except Exception as e:
                        print(f"替代方法失敗: {e}")
                        continue
            
            # 如果找到解離位點
            if dissociable_atoms:
                # 提取索引和官能團類型
                indices = [str(idx) for idx, _ in dissociable_atoms]
                groups = [group for _, group in dissociable_atoms]
                
                dissociable_atoms_str = ':'.join(indices)
                functional_group_str = ':'.join(groups)
                
                # 添加到結果數據框
                result_df = pd.concat([result_df, pd.DataFrame({
                    'SMILES': [smiles],
                    'pKa': [pka],
                    'Dissociable_Atoms': [dissociable_atoms_str],
                    'Functional_Group': [functional_group_str],
                    'Name': [name]
                })], ignore_index=True)
                success += 1
            else:
                print(f"警告: 無法在分子中找到可解離原子: {smiles}, {name}")
                error_log.append(f"無法識別解離位點: {smiles}")
                
                # 最後嘗試: 使用結構分析和字符串搜索
                final_attempt = False
                
                # 檢查是否含有環狀結構
                if ('1' in smiles or '2' in smiles or '3' in smiles) and any(x in smiles for x in ['c', 'n', 'o', 's']):
                    # 這是芳香或雜環化合物，使用SMILES直接分析
                    try:
                        fallback_result = analyze_smiles_directly(smiles)
                        if fallback_result:
                            indices = [str(idx) for idx, _ in fallback_result]
                            groups = [group for _, group in fallback_result]
                            
                            result_df = pd.concat([result_df, pd.DataFrame({
                                'SMILES': [smiles],
                                'pKa': [pka],
                                'Dissociable_Atoms': [':'.join(indices)],
                                'Functional_Group': [':'.join(groups)],
                                'Name': [name + ' (環狀結構估計)']
                            })], ignore_index=True)
                            success += 1
                            final_attempt = True
                    except:
                        pass
                
                # 對於氨基酸結構，直接估算解離位點
                if not final_attempt and 'N' in smiles and ('C(O)=O' in smiles or 'C(=O)O' in smiles):
                    try:
                        # 這是氨基酸結構的基本猜測
                        n_idx = smiles.find('N')
                        cooh_idx = smiles.find('C(O)=O')
                        if cooh_idx == -1:
                            cooh_idx = smiles.find('C(=O)O')
                            if cooh_idx != -1:
                                cooh_idx += 4  # O的位置在C(=O)O中的索引4
                        else:
                            cooh_idx += 2  # O的位置在C(O)=O中的索引2
                        
                        indices = []
                        groups = []
                        
                        if n_idx != -1:
                            indices.append(str(n_idx))
                            groups.append('amine_primary')
                        
                        if cooh_idx != -1:
                            indices.append(str(cooh_idx))
                            groups.append('carboxylic_acid')
                        
                        if indices and groups:
                            result_df = pd.concat([result_df, pd.DataFrame({
                                'SMILES': [smiles],
                                'pKa': [pka],
                                'Dissociable_Atoms': [':'.join(indices)],
                                'Functional_Group': [':'.join(groups)],
                                'Name': [name + ' (估計解離位點)']
                            })], ignore_index=True)
                            success += 1
                            final_attempt = True
                    except:
                        pass
                
                # 最後嘗試：使用原子符號和數字估算位置
                if not final_attempt:
                    # 檢查SMILES中包含的可解離原子
                    possible_atoms = []
                    for i, char in enumerate(smiles):
                        if char in ['O', 'N', 'S']:
                            # 判斷是否可能是可解離位點(例如是否後面跟著H)
                            if i+1 < len(smiles) and smiles[i+1] == 'H':
                                group = 'alcohol' if char == 'O' else 'amine_primary' if char == 'N' else 'thiol'
                                possible_atoms.append((i, group))
                    
                    # 如果找到可能的解離位點
                    if possible_atoms:
                        indices = [str(idx) for idx, _ in possible_atoms]
                        groups = [group for _, group in possible_atoms]
                        
                        result_df = pd.concat([result_df, pd.DataFrame({
                            'SMILES': [smiles],
                            'pKa': [pka],
                            'Dissociable_Atoms': [':'.join(indices)],
                            'Functional_Group': [':'.join(groups)],
                            'Name': [name + ' (字符估計)']
                        })], ignore_index=True)
                        success += 1
                        final_attempt = True
                    
                # 完全失敗的情況
                if not final_attempt:
                    result_df = pd.concat([result_df, pd.DataFrame({
                        'SMILES': [smiles],
                        'pKa': [pka],
                        'Dissociable_Atoms': ['0'],  # 默認第一個原子
                        'Functional_Group': ['unknown'],
                        'Name': [name + ' (無法識別解離位點)']
                    })], ignore_index=True)
                    failed += 1
                    
        except Exception as e:
            error_message = f"處理分子時出錯: {smiles if 'smiles' in locals() else 'unknown'}, {name if 'name' in locals() else 'unknown'}, 錯誤: {e}"
            print(error_message)
            error_log.append(error_message)
            failed += 1
    
    # 保存結果
    try:
        result_df.to_csv(output_file, index=False)
        print(f"轉換完成! 已保存到 {output_file}")
    except Exception as e:
        print(f"保存結果時出錯: {e}")
    
    # 保存錯誤日誌
    log_file = os.path.splitext(output_file)[0] + "_errors.log"
    with open(log_file, "w") as f:
        f.write("\n".join(error_log))
    
    print(f"統計: 處理 {processed} 條記錄, 成功 {success} 條, 失敗 {failed} 條")
    print(f"錯誤日誌已保存到 {log_file}")
    
    return result_df

def determine_functional_group(atom, mol):
    """根據原子及其環境確定官能團類型"""
    symbol = atom.GetSymbol()
    idx = atom.GetIdx()
    
    if symbol == 'O':
        # 檢查是否是羧酸
        c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
        if c_neighbors:
            for c in c_neighbors:
                has_carbonyl = False
                for bond in c.GetBonds():
                    if (bond.GetOtherAtomIdx(c.GetIdx()) != idx and
                        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                        mol.GetAtomWithIdx(bond.GetOtherAtomIdx(c.GetIdx())).GetSymbol() == 'O'):
                        has_carbonyl = True
                        break
                if has_carbonyl:
                    return 'carboxylic_acid'
        
        # 檢查是否是酚
        if any(n.GetIsAromatic() for n in atom.GetNeighbors()):
            return 'phenol'
        
        # 否則是醇
        return 'alcohol'
    
    elif symbol == 'N':
        # 根據氫原子數確定胺類型
        if atom.GetTotalNumHs() >= 2:
            return 'amine_primary'
        elif atom.GetTotalNumHs() == 1:
            # 如果是環狀或有芳香性質，可能是雜環中的N-H
            if atom.IsInRing() or atom.GetIsAromatic():
                return 'amine_secondary'
            else:
                return 'amine_secondary'
        else:
            return 'amine_tertiary'
    
    elif symbol == 'S':
        return 'thiol'
    
    elif symbol == 'P':
        return 'phosphate'
    
    # 默認情況
    return 'unknown'

if __name__ == "__main__":
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description='將NIST數據轉換為模型需要的格式，強化環狀化合物支持')
    parser.add_argument('--input', required=True, help='輸入NIST數據文件路徑')
    parser.add_argument('--output', default='OMGNN/data/nist_pka_data.csv', help='輸出文件路徑')
    parser.add_argument('--verbose', action='store_true', help='顯示詳細日誌')
    
    args = parser.parse_args()
    
    # 設置詳細日誌
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # 執行轉換
    convert_nist_data(args.input, args.output)
