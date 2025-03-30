import torch
import argparse
import os
import pandas as pd
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from pka_models import PKA_GNN
from pka_datautils import PKADataProcessor
from pka_chemutils import tensorize_molecule, predict_dissociable_atoms

"""
1. 載入參數
"""
if __name__ == '__main__':
    """超參數存在parameters.csv中,
    透過--version指定使用的版本,方便記錄使用的參數設定"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_csv', default='../data/pka_parameters.csv',
                        help='配置文件路徑')   
    parser.add_argument('--version', default='pka_gnn_v1',
                        help='配置版本，例如: v1, v2')
    parser.add_argument('--input_data', default='../data/nist_pka_data_clean.csv',
                        help='輸入數據文件路徑')
    parser.add_argument('--smiles_col', default='SMILES',
                        help='SMILES列名')
    parser.add_argument('--pka_col', default='pKa',
                        help='pKa值列名')
    parser.add_argument('--dissociable_atom_col', default=None,
                        help='可解離原子索引列名（可選）')
    args = parser.parse_args()
    
    # 加載配置
    try:
        config = PKADataProcessor.load_config_by_version(args.config_csv, args.version)
    except FileNotFoundError:
        # 如果找不到配置文件，使用默認配置
        config = {
            'version': args.version,
            'test_size': 0.2,
            'num_features': 153,  # 根據atom_features函數的輸出維度
            'bond_features': 12,  # 根據bond_features函數的輸出維度
            'hidden_dim': 128,
            'batch_size': 32,
            'num_epochs': 100,
            'lr': 0.001,
            'dropout': 0.2,
            'anneal_rate': 0.9,
            'save_path': '../results/',
            'model_path': '../models/'
        }
        # 創建配置文件目錄
        os.makedirs(os.path.dirname(args.config_csv), exist_ok=True)
        
        # 將默認配置保存到配置文件
        pd.DataFrame([config]).to_csv(args.config_csv, index=False)
        print(f"找不到配置文件，已創建默認配置文件: {args.config_csv}")
    
    # 確保結果和模型目錄存在
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(config['model_path'], exist_ok=True)
    
    # 打印配置參數
    print("配置參數:")
    for k, v in config.items():
        print(f"{k}: {v}")

"""
2. 載入資料
"""
# 加載數據
train_loader, test_loader = PKADataProcessor.data_loader(
    file_path=args.input_data,
    smiles_col=args.smiles_col,
    pka_col=args.pka_col,
    dissociable_atom_col=args.dissociable_atom_col,
    batch_size=config['batch_size'],
    test_size=config['test_size']
)

# 設置計算設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

# 創建模型
model = PKA_GNN(
    node_dim=config['num_features'],
    bond_dim=config['bond_features'],
    hidden_dim=config['hidden_dim'],
    dropout=config['dropout']
).to(device)

# 設置優化器和學習率調度器
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['anneal_rate'])

"""
3. 開始訓練
"""
# 初始化歷史記錄
train_loss_history = []
train_cls_history = []
train_reg_history = []
train_acc_history = []
test_loss_history = []
test_cls_history = []
test_reg_history = []
test_acc_history = []

# 訓練循環
for epoch in range(config['num_epochs']):
    model.train()
    total_loss, total_cls_loss, total_reg_loss = 0, 0, 0
    total_accuracy = 0
    batch_count = 0
    
    # 訓練階段
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        
        # 前向傳播
        _, _, losses = model(batch)
        all_loss, cls_loss, reg_loss, accuracy = losses
        
        # 反向傳播和優化
        all_loss.backward()
        optimizer.step()
        
        # 累加損失和準確率
        total_loss += all_loss.item()
        total_cls_loss += cls_loss.item()
        # 檢查reg_loss是否為張量，如果是則使用item()，否則直接加
        total_reg_loss += reg_loss.item() if hasattr(reg_loss, 'item') else reg_loss
        total_accuracy += accuracy
        batch_count += 1
    # 計算平均損失和準確率
    train_loss = total_loss / batch_count
    train_cls_loss = total_cls_loss / batch_count
    train_reg_loss = total_reg_loss / batch_count
    train_accuracy = total_accuracy / batch_count
    
    # 更新學習率
    scheduler.step()
    
    # 評估階段
    model.eval()
    with torch.no_grad():
        test_cls_acc, test_reg_mse, test_results = PKADataProcessor.evaluate_model(
            model, test_loader, device)
        
        # 計算額外的評估指標
        if len(test_results['True_pKa']) > 0 and len(test_results['Pred_pKa']) > 0:
            mae = mean_absolute_error(test_results['True_pKa'], test_results['Pred_pKa'])
            r2 = r2_score(test_results['True_pKa'], test_results['Pred_pKa']) if len(test_results['True_pKa']) > 1 else 0
            print(f"測試集 - MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # 計算總測試損失
    test_loss = test_cls_acc + test_reg_mse
    
    # 記錄歷史
    train_loss_history.append(train_loss)
    train_cls_history.append(train_cls_loss)
    train_reg_history.append(train_reg_loss)
    train_acc_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_cls_history.append(test_cls_acc)
    test_reg_history.append(test_reg_mse)
    test_acc_history.append(test_cls_acc)
    
    # 打印進度
    print(f"Epoch {epoch+1}/{config['num_epochs']}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_cls_acc:.4f}, Test MSE: {test_reg_mse:.4f}")
    
    # 保存訓練歷史
    history_df = pd.DataFrame({
        'train_loss': train_loss_history,
        'train_cls_loss': train_cls_history,
        'train_reg_loss': train_reg_history,
        'train_accuracy': train_acc_history,
        'test_loss': test_loss_history,
        'test_cls_acc': test_cls_history,
        'test_reg_mse': test_reg_history,
        'test_accuracy': test_acc_history
    })
    history_df.to_csv(
        os.path.join(config['save_path'], f"{config['version']}_training_history.csv"), 
        index=False
    )
    
    # 每10個epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(config['model_path'], f"{config['version']}_epoch{epoch+1}.pt")
        )

"""
4. 評估模型
"""
# 在整個訓練和測試集上進行最終評估
print("\n最終評估:")

# 訓練集評估
train_cls_acc, train_reg_mse, train_results = PKADataProcessor.evaluate_model(
    model, train_loader, device,
    output_path=os.path.join(config['save_path'], f"{config['version']}_final_train_predictions.csv")
)
print(f"訓練集 - 分類準確率: {train_cls_acc:.4f}, 回歸MSE: {train_reg_mse:.4f}")

# 測試集評估
test_cls_acc, test_reg_mse, test_results = PKADataProcessor.evaluate_model(
    model, test_loader, device,
    output_path=os.path.join(config['save_path'], f"{config['version']}_final_test_predictions.csv")
)
print(f"測試集 - 分類準確率: {test_cls_acc:.4f}, 回歸MSE: {test_reg_mse:.4f}")

# 保存最終模型
torch.save(
    model.state_dict(),
    os.path.join(config['model_path'], f"{config['version']}_final.pt")
)
print(f"最終模型已保存至: {os.path.join(config['model_path'], config['version'] + '_final.pt')}")

print("\n訓練完成!") 