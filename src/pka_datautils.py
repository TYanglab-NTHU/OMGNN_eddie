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
        """
        設置是否顯示詳細日誌
        
        參數:
            enable (bool): 是否啟用詳細日誌
        """
        PKADataProcessor.verbose = enable
    
    @staticmethod
    def log(message):
        """
        根據verbose設置輸出日誌
        
        參數:
            message (str): 日誌訊息
        """
        if PKADataProcessor.verbose:
            print(message)
    
    @staticmethod
    def load_config_by_version(csv_path, version):
        """
        從CSV配置文件中加載特定版本的配置參數。
        
        參數:
            csv_path (str): 配置文件路徑
            version (str): 配置版本
        
        返回:
            dict: 配置參數字典
        """
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
    def data_loader(file_path, smiles_col='SMILES', pka_col='pKa', dissociable_atom_col=None, batch_size=32, test_size=0.2):
        """
        加載pKa數據並處理為PyTorch Geometric數據集。
        
        參數:
            file_path (str): 數據文件路徑（CSV格式）
            smiles_col (str): SMILES列名
            pka_col (str): pKa值列名
            dissociable_atom_col (str): 可解離原子索引列名（可選）
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
        
        # 分割訓練集和測試集
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # 轉換數據為Geometric數據集
        train_dataset = PKADataProcessor._create_dataset(train_df, smiles_col, pka_col, dissociable_atom_col)
        test_dataset = PKADataProcessor._create_dataset(test_df, smiles_col, pka_col, dissociable_atom_col)
        
        # 創建數據加載器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    @staticmethod
    def _create_dataset(df, smiles_col, pka_col, dissociable_atom_col):
        """
        創建PyTorch Geometric數據集。
        
        參數:
            df (DataFrame): 數據框
            smiles_col (str): SMILES列名
            pka_col (str): pKa值列名
            dissociable_atom_col (str): 可解離原子索引列名（可選）
        
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
                except:
                    # 如果解析失敗，使用integrated_nist_converter中的函數預測可解離原子
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
            
            # 為每個可解離原子創建pKa值對
            pka_values = [(idx, pka) for idx in dissociable_indices]
            
            # 將分子轉換為圖表示
            try:
                x, edge_index, edge_attr, dissociable_masks, pka_tensor, mol = tensorize_molecule(
                    smiles, dissociable_indices, pka_values)
                
                if x is not None and edge_index is not None:
                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        dissociable_masks=dissociable_masks,
                        pka_values=pka_tensor,
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
        評估模型性能。
        
        參數:
            model: 要評估的模型
            loader: 數據加載器
            device: 計算設備
            output_path: 結果輸出路徑（可選）
        
        返回:
            tuple: (classification_accuracy, regression_mse, results_df)
        """
        model.eval()
        
        # 使用列表的列表來確保數據同步
        results = []
        
        correct_cls = 0
        total_cls = 0
        mse_loss = 0
        total_pka_predictions = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                
                # 獲取預測結果
                cls_preds, reg_preds, _ = model(data)
                
                # 處理分類結果
                true_dissociable = data.dissociable_masks.bool()
                pred_dissociable = torch.tensor([pred == 1 for pred in cls_preds], device=device)
                
                # 計算分類準確率
                correct_cls += (pred_dissociable == true_dissociable).sum().item()
                total_cls += len(true_dissociable)
                
                # 處理回歸結果 - 確保數據同步
                for idx, pred_pka in reg_preds:
                    if idx < len(data.pka_values) and not torch.isnan(data.pka_values[idx]):
                        true_pka = data.pka_values[idx].item()
                        
                        # 將所有相關數據作為一個元組存儲，確保同步
                        if hasattr(data, 'smiles'):
                            smiles = data.smiles
                            results.append((smiles, true_pka, pred_pka))
                        
                        # 計算MSE時添加保護措施
                        try:
                            # 防止極端值
                            if np.isfinite(pred_pka) and np.isfinite(true_pka):
                                mse_loss += (true_pka - pred_pka) ** 2
                                total_pka_predictions += 1
                        except:
                            if PKADataProcessor.verbose:
                                print(f"警告：pKa計算出現問題，true_pka={true_pka}, pred_pka={pred_pka}")
        
        # 計算最終指標
        cls_accuracy = correct_cls / total_cls if total_cls > 0 else 0
        reg_mse = mse_loss / total_pka_predictions if total_pka_predictions > 0 else float('inf')
        
        # 從結果列表中提取數據，確保長度一致
        all_smiles, all_true_pka, all_pred_pka = [], [], []
        for smiles, true_pka, pred_pka in results:
            all_smiles.append(smiles)
            all_true_pka.append(true_pka)
            all_pred_pka.append(pred_pka)
        
        # 創建結果數據框 - 現在我們確保所有列表的長度相同
        if all_smiles and len(all_smiles) == len(all_true_pka) == len(all_pred_pka):
            results_df = pd.DataFrame({
                'SMILES': all_smiles,
                'True_pKa': all_true_pka,
                'Pred_pKa': all_pred_pka
            })
        else:
            # 如果數據不一致，創建一個空的DataFrame並記錄錯誤
            PKADataProcessor.log(f"警告：數據長度不一致，SMILES: {len(all_smiles)}, True_pKa: {len(all_true_pka)}, Pred_pKa: {len(all_pred_pka)}")
            results_df = pd.DataFrame()
        
        # 如果指定了輸出路徑，保存結果
        if output_path and not results_df.empty:
            results_df.to_csv(output_path, index=False)
        
        return cls_accuracy, reg_mse, results_df 