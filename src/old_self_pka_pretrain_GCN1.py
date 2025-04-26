import torch
import torch.nn as nn
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader
import sys
sys.path.append('src/')
from self_pka_trainutils import load_config_by_version, pka_Dataloader
import importlib
import self_pka_models
# importlib.reload(self_pka_models)
from self_pka_models import pka_GNN
from self_pka_chemutils import tensorize_for_pka
import pandas as pd
import os
import numpy as np
import time
from pka_learning_curve import plot_learning_curves

# 引入異常檢測功能
from torch.autograd import set_detect_anomaly
# load parameters
if __name__ == '__main__':
    """超參數存在parameters.csv中,
    透過--version指定使用的版本,方便記錄使用的參數設定"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_csv', default='../data/pka_parameters.csv',
                        help='path to pka_parameters.csv')   
    parser.add_argument('--version'   , default='pka_ver1',
                        help='e.g., v1, v2')
    args   = parser.parse_args()
    config = load_config_by_version(args.config_csv, args.version)
    # config = load_config_by_version('../data/pka_parameters.csv', 'pka_ver1')

    for k, v in config.items():
        # 打印配置參數，使用固定寬度格式以確保整齊對齊
        print(f"{k.ljust(15)}: {v}")

    # 資料載入
    # 設定所需的欄位
    columns = ['smiles', 'pka_values', 'pka_matrix']
    
    # 使用新的pka_dataloader載入資料
    train_loader, test_loader = pka_Dataloader.data_loader_pka(
        file_path=config['input'],
        columns=columns,
        tensorize_fn=tensorize_for_pka,
        batch_size=config['batch_size'],
        test_size=config['test_size'],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建pKa預測模型
    model = pka_GNN(
        node_dim=config['num_features'], 
        bond_dim=11, 
        hidden_dim=config['num_features'], 
        output_dim=1,  # 每個原子的pKa值輸出
        dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # 配置學習率預熱和衰減
    def lr_lambda(epoch):
        # 前5個epoch預熱
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # 線性增長到設定的學習率
        else:
            # 之後使用餘弦衰減
            decay_rate = config['anneal_rate']
            return decay_rate ** (epoch - warmup_epochs)
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # 原本這邊有class和reg的損失
    
    # 設定梯度裁剪閾值，防止梯度爆炸
    grad_clip_value = 1.0
    
    # 創建一個空的DataFrame來儲存所有訓練和測試的損失值
    df_loss = pd.DataFrame(columns=[
        'epoch', 'train_loss', 'train_cla_loss', 'train_reg_loss', 
        'test_loss', 'test_cla_loss', 'test_reg_loss', 'test_classification_metrics'
    ])

    best_test_loss = float('inf')
    best_model_state = None

    start_time = time.time()
    # 啟用PyTorch自動梯度異常檢測功能，幫助定位就地操作問題
    set_detect_anomaly(False)
    print("已啟用PyTorch梯度異常檢測，這會減慢訓練速度但能幫助定位問題")
    
    # ==============訓練階段==============
    for epoch in range(config['num_epochs']):
        # 打印當前學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, 當前學習率: {current_lr:.6f}")
        
        model.train()
        train_loss = 0
        train_cla_loss = 0
        train_reg_loss = 0
        batch_count = 0
        
        # 追蹤NaN損失的批次數
        nan_loss_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            
            # 獲取模型預測與損失
            class_logits, pka_values, losses = model(batch)
            loss, loss_cla, loss_reg = losses
            
            # 檢查損失值是否為NaN，如果是則跳過此批次
            if torch.isnan(loss) or torch.isnan(loss_cla) or torch.isnan(loss_reg):
                print(f"警告：檢測到NaN損失值! loss={loss.item() if not torch.isnan(loss) else 'NaN'}, "
                      f"loss_cla={loss_cla.item() if not torch.isnan(loss_cla) else 'NaN'}, "
                      f"loss_reg={loss_reg.item() if not torch.isnan(loss_reg) else 'NaN'}")
                
                # 檢查輸入數據是否有問題
                pka_mask = batch.pka_labels > 0
                if pka_mask.any():
                    true_pka = batch.pka_labels[pka_mask]
                    pred_pka = pka_values[pka_mask]
                    print(f"輸入檢查 - true_pka: {true_pka}, pred_pka: {pred_pka}")
                
                # 計數NaN批次
                nan_loss_batches += 1
                
                # 跳過此批次
                continue
            
            # 反向傳播與優化
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"反向傳播出錯: {e}")
                print(f"出錯的批次SMILES: {batch.smiles}")
                print(f"出錯的批次pka標籤: {batch.pka_labels}")
                
                # 清除梯度並跳過此批次
                optimizer.zero_grad()
                continue
            
            # 應用梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            # 檢查梯度是否包含NaN或Inf
            has_nan_or_inf = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_or_inf = True
                        break
            
            if has_nan_or_inf:
                print("警告：檢測到NaN或Inf梯度，跳過參數更新！")
                # 清除梯度
                optimizer.zero_grad()
                continue
                
            optimizer.step()
            
            # 累計損失 - 分別處理每個損失值
            train_loss += loss.item()
            train_cla_loss += loss_cla.item()
            train_reg_loss += loss_reg.item()
            
            batch_count += 1
        
        # 計算平均訓練損失
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_train_cla_loss = train_cla_loss / batch_count if batch_count > 0 else 0
        avg_train_reg_loss = train_reg_loss / batch_count if batch_count > 0 else 0
        
        # 打印NaN批次統計
        if nan_loss_batches > 0:
            print(f"本輪訓練中有 {nan_loss_batches} 個批次出現NaN損失值被跳過")
        
        # 測試階段 - 檢查一下evaluate_pka_model函數的定義
        try:
            test_loss, test_cla_loss, test_reg_loss, test_classification_metrics = pka_Dataloader.evaluate_pka_model(
                model, 
                test_loader, 
                device, 
                output_file=f"{config['version']}_epoch_{epoch}_test.csv",
                save_path=config['save_path']
            )
        except Exception as e:
            print(f"評估過程中出錯: {e}")
            # 如果函數有問題，返回虛擬值
            test_loss, test_cla_loss, test_reg_loss, test_classification_metrics = 0, 0, 0, 0
        
        # 打印訓練進度
        print(f"Epoch [train] {epoch+1}/{config['num_epochs']}, "
              f"Train Loss: {avg_train_loss:.4f} (Cla: {avg_train_cla_loss:.4f}, Reg: {avg_train_reg_loss:.4f}), "
              f"Test Loss: {test_loss:.4f} (Cla: {test_cla_loss:.4f}, Reg: {test_reg_loss:.4f}), "
              f"Test Cls Acc: {test_classification_metrics['pka_atom_accuracy']:.4f}")
        # 保存當前epoch的損失值
        df_loss.loc[epoch] = [
            epoch, avg_train_loss, avg_train_cla_loss, avg_train_reg_loss,
            test_loss, test_cla_loss, test_reg_loss, test_classification_metrics
        ]
        
        # 保存DataFrame到CSV文件
        df_loss.to_csv(os.path.join(config['save_path'], f"{config['version']}_loss.csv"), index=False)
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(
                model.state_dict(), 
                os.path.join(config['model_path'], f"{config['version']}_best.pkl")
            )
        
        # 更新學習率
        scheduler.step()

    # ==============訓練結束==============
    print(f"訓練結果已經保存到 {config['save_path']}{config['version']}_loss.csv")
    end_time = time.time()
    print(f"訓練時間: {end_time - start_time:.2f} 秒")

    # 保存最後一個epoch的模型
    torch.save(
        model.state_dict(), 
        os.path.join(config['model_path'], f"{config['version']}_final.pkl")
    )
    
    # 如果有最佳模型，載入它並進行最終評估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        final_test_loss, final_test_cla_loss, final_test_reg_loss, test_classification_metrics = pka_Dataloader.evaluate_pka_model(
            model, 
            test_loader, 
            device, 
            output_file=f"{config['version']}_best_final.csv"
        )
        print(f"最佳模型 - Test Loss: {final_test_loss:.4f} "
              f"(Cla: {final_test_cla_loss:.4f}, Reg: {final_test_reg_loss:.4f}), "
              f"Test Cls Acc: {test_classification_metrics['pka_atom_accuracy']:.4f}")
    # 畫學習曲線
    plot_learning_curves(
        csv_path=os.path.join(config['save_path'], f"{config['version']}_loss.csv"),
        output_dir=config['save_path'],
        version_name=config['version']
    )
    
    
    is_Sample = False
    if is_Sample:
        # 確保模型處於評估模式
        model.eval()
        
        # 導入用於分析的模組
        # 注意這裡使用延遲導入，避免循環導入的問題
        from self_pka_sample import split_and_sample_dataset
        
        # 在訓練結束後執行數據分割與採樣
        print("\n開始執行數據分割與採樣...")
        # 將當前訓練好的模型傳遞給split_and_sample_dataset函數
        split_and_sample_dataset(model=model, device=device)
        print("數據分割與採樣完成！")
        
        # 生成parity plot
        print("\n開始生成parity plot...")
        from plot_pka_parity import plot_train_test_comparison, plot_single_dataset, combine_csv_files
        
        # 繪製訓練集和測試集的對比圖
        plot_train_test_comparison()
        
        combine_csv_files(
            train_file="../output/pka_train_prediction_results.csv",
            test_file="../output/pka_test_prediction_results.csv",
            output_file="../output/pka_all_prediction_results.csv"
        )
        
        print("parity plot生成完成！")
        