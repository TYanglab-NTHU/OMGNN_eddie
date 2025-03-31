import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from integrated_nist_converter import predict_dissociable_atoms, sanitize_smiles, tensorize_molecule, set_verbose
set_verbose(False)

class PKADataProcessor:
    """處理pKa預測的數據類"""
    
    # 添加靜態變數控制是否顯示詳細日誌
    verbose = False
    
    @staticmethod
    def set_verbose(enable=True):
        """設置是否顯示詳細日誌"""
        PKADataProcessor.verbose = enable
    
    @staticmethod
    def log(message):
        """根據verbose設置輸出日誌"""
        if PKADataProcessor.verbose:
            print(message)
    
    @staticmethod
    def load_config_by_version(csv_path, version):
        """從CSV配置文件中加載特定版本的配置參數"""
        df = pd.read_csv(csv_path)
        # 獲取指定版本的配置
        config_row = df[df['version'] == version]
        if config_row.empty:
            raise ValueError(f"找不到版本 {version} 的配置")
        config = config_row.iloc[0].to_dict()
        
        # 轉換參數為適當的類型
        config['test_size'] = float(config['test_size'])
        config['num_features'] = int(config['num_features'])
        config['batch_size'] = int(config['batch_size'])
        config['num_epochs'] = int(config['num_epochs'])
        config['dropout'] = float(config['dropout'])
        
        return config
    
    @staticmethod
    def data_loader(file_path, smiles_col='SMILES', pka_col='pKa', 
                   dissociable_atom_col='Dissociable_Atoms_Ordered', 
                   func_group_col='Functional_Group_Ordered',
                   batch_size=32, test_size=0.2):
        """
        加載pKa數據並處理為PyTorch Geometric數據集，支持解離順序
        
        參數:
            file_path (str): 數據文件路徑（CSV格式）
            smiles_col (str): SMILES列名
            pka_col (str): pKa值列名
            dissociable_atom_col (str): 可解離原子索引列名（優先使用有序的列）
            func_group_col (str): 官能團類型列名（優先使用有序的列）
            batch_size (int): 批次大小
            test_size (float): 測試集比例
        
        返回:
            tuple: (train_loader, test_loader)
        """
        # 加載數據
        df = pd.read_csv(file_path)
        
        # 檢查必要列
        if smiles_col not in df.columns:
            raise ValueError(f"找不到SMILES列：{smiles_col}")
        if pka_col not in df.columns:
            raise ValueError(f"找不到pKa列：{pka_col}")
        
        # 如果找不到有序的解離原子列，則使用原始列
        if dissociable_atom_col not in df.columns:
            dissociable_atom_col = 'Dissociable_Atoms'
            PKADataProcessor.log(f"找不到有序解離原子列，使用原始列: {dissociable_atom_col}")
        
        # 分割訓練集和測試集
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # 轉換數據為Geometric數據集
        train_dataset = PKADataProcessor._create_dataset(
            train_df, smiles_col, pka_col, dissociable_atom_col, func_group_col)
        test_dataset = PKADataProcessor._create_dataset(
            test_df, smiles_col, pka_col, dissociable_atom_col, func_group_col)
        
        # 創建數據加載器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    @staticmethod
    def _create_dataset(df, smiles_col, pka_col, dissociable_atom_col, func_group_col=None):
        """
        創建PyTorch Geometric數據集，支持解離順序
        
        參數:
            df (DataFrame): 數據框
            smiles_col (str): SMILES列名
            pka_col (str): pKa值列名
            dissociable_atom_col (str): 可解離原子索引列名
            func_group_col (str): 官能團類型列名
        
        返回:
            list: PyTorch Geometric數據對象列表
        """
        dataset = []
        
        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            pka = row[pka_col]
            
            # 嘗試修復SMILES語法問題
            smiles = sanitize_smiles(smiles)
            
            # 獲取可解離原子索引
            if dissociable_atom_col and dissociable_atom_col in df.columns:
                # 如果數據中已提供可解離原子索引
                try:
                    # 處理可能的多種分隔符格式
                    dissociable_str = str(row[dissociable_atom_col])
                    if ':' in dissociable_str:
                        # 如果使用冒號分隔，如 integrated_nist_converter 輸出的格式
                        dissociable_indices = [int(x) for x in dissociable_str.split(':') if x.strip()]
                    else:
                        # 如果使用逗號分隔
                        dissociable_indices = [int(x) for x in dissociable_str.split(',') if x.strip()]
                except Exception as e:
                    PKADataProcessor.log(f"解析解離原子索引時出錯: {e}, 使用預測的可解離原子")
                    try:
                        dissociable_atoms = predict_dissociable_atoms(smiles)
                        dissociable_indices = [idx for idx, _ in dissociable_atoms]
                    except Exception as e:
                        PKADataProcessor.log(f"無法處理SMILES: {smiles}, 錯誤: {e}")
                        continue
            else:
                # 使用integrated_nist_converter中的函數預測可解離原子
                try:
                    dissociable_atoms = predict_dissociable_atoms(smiles)
                    dissociable_indices = [idx for idx, _ in dissociable_atoms]
                except Exception as e:
                    PKADataProcessor.log(f"無法處理SMILES: {smiles}, 錯誤: {e}")
                    continue
            
            # 如果沒有找到可解離原子，繼續下一個分子
            if not dissociable_indices:
                PKADataProcessor.log(f"警告: 在分子 {smiles} 中未找到可解離原子，跳過此分子")
                continue
            
            # 獲取官能團信息
            func_groups = []
            if func_group_col and func_group_col in df.columns:
                try:
                    func_group_str = str(row[func_group_col])
                    func_groups = func_group_str.split(':')
                except Exception as e:
                    PKADataProcessor.log(f"解析官能團信息時出錯: {e}")
            
            # 確保官能團數量與解離原子數量匹配
            if func_groups and len(func_groups) != len(dissociable_indices):
                PKADataProcessor.log(f"警告: 官能團數量({len(func_groups)})與解離原子數量({len(dissociable_indices)})不匹配")
                func_groups = func_groups[:len(dissociable_indices)] if len(func_groups) > len(dissociable_indices) else func_groups + ['unknown'] * (len(dissociable_indices) - len(func_groups))
            
            # 根據pKa值推斷當前解離位點
            current_dissociation_idx = 0
            if len(dissociable_indices) > 1:
                # 使用pKa值推斷當前解離位點
                # 假設低pKa值(酸性環境)時，先解離羧酸等酸性基團
                # 高pKa值(鹼性環境)時，解離胺基等鹼性基團
                if pka < 7.0:  # 假設7.0為分界點
                    current_dissociation_idx = 0  # 酸性基團通常排在前面
                else:
                    current_dissociation_idx = min(1, len(dissociable_indices) - 1)  # 鹼性基團通常排在后面
            
            # 將解離順序信息編碼為一個列表，表示每個位點的解離順序（從0開始）
            dissociation_order = list(range(len(dissociable_indices)))
            
            # 為每個可解離原子創建pKa值對，包括解離順序信息
            pka_values_with_order = [(idx, pka, order) for idx, order in zip(dissociable_indices, dissociation_order)]
            
            # 將分子轉換為圖表示
            try:
                x, edge_index, edge_attr, dissociable_masks, pka_tensor, mol = tensorize_molecule(
                    smiles, dissociable_indices, [(idx, pka) for idx, pka, _ in pka_values_with_order])
                
                # 創建解離順序張量
                dissociation_order_tensor = torch.full((len(x),), -1, dtype=torch.long)
                for atom_idx, _, order in pka_values_with_order:
                    dissociation_order_tensor[atom_idx] = order
                
                # 創建當前解離位點張量
                current_dissociation_tensor = torch.zeros((len(x),), dtype=torch.bool)
                if dissociable_indices:
                    current_dissociation_tensor[dissociable_indices[current_dissociation_idx]] = True
                
                if x is not None and edge_index is not None:
                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        dissociable_masks=dissociable_masks,
                        pka_values=pka_tensor,
                        dissociation_order=dissociation_order_tensor,
                        current_dissociation=current_dissociation_tensor,
                        smiles=smiles
                    )
                    dataset.append(data)
                else:
                    PKADataProcessor.log(f"警告: 無法將分子 {smiles} 轉換為圖表示")
            except Exception as e:
                PKADataProcessor.log(f"處理分子 {smiles} 時出錯: {e}")
                continue
        
        return dataset
    
    @staticmethod
    def evaluate_model(model, loader, device, output_path=None):
        """
        評估模型性能，支持解離順序評估
        
        參數:
            model: 要評估的模型
            loader: 數據加載器
            device: 計算設備
            output_path: 結果輸出路徑（可選）
        
        返回:
            tuple: (classification_accuracy, regression_mse, results_df)
        """
        model.eval()
        
        # 初始化結果存儲
        all_smiles = []
        all_true_pka = []
        all_pred_pka = []
        all_true_labels = []
        all_pred_labels = []
        all_dissociation_orders = []  # 新增：解離順序
        all_functional_groups = []    # 新增：官能團信息
        
        correct_cls = 0
        total_cls = 0
        mse_loss = 0
        total_pka_predictions = 0
        
        # 分子級別的結果聚合
        molecule_results = {}
        
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(device)
                
                # 獲取預測結果，包括解離順序
                cls_preds, reg_preds, _ = model(data)
                
                # 處理分類結果
                true_dissociable = data.dissociable_masks.bool()
                pred_dissociable = torch.tensor([pred == 1 for pred in cls_preds], device=device)
                
                # 計算分類準確率
                correct_cls += (pred_dissociable == true_dissociable).sum().item()
                total_cls += len(true_dissociable)
                
                # 保存分類結果用於F1計算
                all_true_labels.extend(true_dissociable.cpu().numpy().tolist())
                all_pred_labels.extend(pred_dissociable.cpu().numpy().tolist())
                
                # 檢查 batch 是否包含 SMILES
                batch_has_smiles = hasattr(data, 'smiles')
                # 檢查 batch 是否包含解離順序
                has_dissociation_order = hasattr(data, 'dissociation_order')
                
                # 處理回歸結果 - 逐個節點處理
                for node_idx, pred_info in enumerate(reg_preds):
                    # 確保 pred_info 是一個包含 node_idx 和 pred_pka 的元組
                    if isinstance(pred_info, tuple) and len(pred_info) >= 2:
                        pred_idx, pred_pka = pred_info[0], pred_info[1]
                    else:
                        # 可能只返回了pKa值而沒有索引
                        pred_idx, pred_pka = node_idx, pred_info
                    
                    # 確保pred_idx是一個整數
                    if isinstance(pred_idx, torch.Tensor):
                        pred_idx = pred_idx.item()
                    
                    # 檢查是否有有效的pKa值
                    if pred_idx < len(data.pka_values) and not torch.isnan(data.pka_values[pred_idx]):
                        true_pka = data.pka_values[pred_idx].item()
                        
                        # 獲取解離順序信息
                        dissociation_order = -1
                        if has_dissociation_order and pred_idx < len(data.dissociation_order):
                            dissociation_order = data.dissociation_order[pred_idx].item()
                        
                        # 將預測結果與對應的 SMILES 配對
                        curr_smiles = "UNKNOWN"
                        if batch_has_smiles:
                            if isinstance(data.smiles, list) and len(data.smiles) > 0:
                                # 使用批次中的第一個SMILES作為後備
                                curr_smiles = data.smiles[0]
                                
                                # 如果可能，尋找正確的分子SMILES
                                if hasattr(data, 'batch') and data.batch is not None:
                                    batch_indices = data.batch.cpu().numpy()
                                    if pred_idx < len(batch_indices):
                                        mol_idx = batch_indices[pred_idx]
                                        if mol_idx < len(data.smiles):
                                            curr_smiles = data.smiles[mol_idx]
                            else:
                                curr_smiles = str(data.smiles)
                        
                        # 添加到結果列表
                        all_smiles.append(curr_smiles)
                        all_true_pka.append(true_pka)
                        all_pred_pka.append(pred_pka)
                        all_dissociation_orders.append(dissociation_order)
                        
                        # 按分子聚合結果
                        if curr_smiles not in molecule_results:
                            molecule_results[curr_smiles] = {
                                'smiles': curr_smiles,
                                'pkas': [],
                                'orders': [],
                                'predictions': []
                            }
                        
                        molecule_results[curr_smiles]['pkas'].append(true_pka)
                        molecule_results[curr_smiles]['orders'].append(dissociation_order)
                        molecule_results[curr_smiles]['predictions'].append(pred_pka)
                        
                        # 計算 MSE 時添加保護措施
                        try:
                            # 防止極端值
                            if np.isfinite(pred_pka) and np.isfinite(true_pka):
                                mse_loss += (true_pka - pred_pka) ** 2
                                total_pka_predictions += 1
                        except Exception as e:
                            if PKADataProcessor.verbose:
                                print(f"警告：pKa計算出現問題，true_pka={true_pka}, pred_pka={pred_pka}, 錯誤: {str(e)}")
        
        # 計算最終指標
        cls_accuracy = correct_cls / total_cls if total_cls > 0 else 0
        reg_mse = mse_loss / total_pka_predictions if total_pka_predictions > 0 else float('inf')
        
        # 創建兩種結果數據框：原子級別和分子級別
        # 原子級別結果
        atom_results_dict = {
            'SMILES': all_smiles,
            'True_pKa': all_true_pka,
            'Pred_pKa': all_pred_pka,
            'Dissociation_Order': all_dissociation_orders,
        }
        
        # 分子級別結果
        molecule_rows = []
        for smiles, data in molecule_results.items():
            # 按解離順序排序結果
            sorted_data = sorted(zip(data['orders'], data['pkas'], data['predictions']), 
                                key=lambda x: x[0] if x[0] >= 0 else float('inf'))
            
            if not sorted_data:
                continue
                
            # 創建分子級別的行
            row = {
                'SMILES': smiles,
                'True_pKa_Values': ';'.join(map(str, [p for _, p, _ in sorted_data])),
                'Pred_pKa_Values': ';'.join(map(str, [pr for _, _, pr in sorted_data])),
                'Dissociation_Orders': ';'.join(map(str, [o for o, _, _ in sorted_data])),
                'Avg_True_pKa': np.mean([p for _, p, _ in sorted_data]),
                'Avg_Pred_pKa': np.mean([pr for _, _, pr in sorted_data]),
                'MSE': np.mean([(p - pr) ** 2 for _, p, pr in sorted_data if np.isfinite(p) and np.isfinite(pr)])
            }
            molecule_rows.append(row)
        
        # 確保所有列表長度一致
        min_len = min(len(v) for v in atom_results_dict.values() if v)
        for k in atom_results_dict:
            if atom_results_dict[k] and len(atom_results_dict[k]) > min_len:
                atom_results_dict[k] = atom_results_dict[k][:min_len]
        
        # 創建結果數據框
        atom_results_df = pd.DataFrame(atom_results_dict) if all_smiles else pd.DataFrame()
        molecule_results_df = pd.DataFrame(molecule_rows) if molecule_rows else pd.DataFrame()
        
        # 如果指定了輸出路徑，保存結果
        if output_path and not atom_results_df.empty:
            # 保存原子級別結果
            atom_output_path = output_path.replace('.csv', '_atom_level.csv')
            atom_results_df.to_csv(atom_output_path, index=False)
            PKADataProcessor.log(f"原子級別結果已保存至: {atom_output_path}, 共 {len(atom_results_df)} 條記錄")
            
            # 保存分子級別結果
            if not molecule_results_df.empty:
                molecule_output_path = output_path.replace('.csv', '_molecule_level.csv')
                molecule_results_df.to_csv(molecule_output_path, index=False)
                PKADataProcessor.log(f"分子級別結果已保存至: {molecule_output_path}, 共 {len(molecule_results_df)} 條記錄")
        
        return cls_accuracy, reg_mse, {"atom_results": atom_results_dict, "molecule_results": molecule_rows} 