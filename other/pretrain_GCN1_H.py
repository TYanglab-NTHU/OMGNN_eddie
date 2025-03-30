import torch
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader 
from datautils  import *
from trainutils import *
from chemutils import *
from models import *
from models import OrganicMetal_GNN



"""
1. 載入參數
"""
if __name__ == '__main__':
    """超參數存在parameters.csv中,
    透過--version指定使用的版本,方便記錄使用的參數設定"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_csv', default='parameters.csv',
                        help='path')   
    parser.add_argument('--version'   , default='rmduplicate_pretrain_OMGNN',
                        help='e.g., v1, v2')
    args   = parser.parse_args()
    config = load_config_by_version(args.config_csv, args.version)

    for k, v in config.items():
        print(k, v)

"""
2. 載入資料
"""
organic_train_data, orgainc_test_loader = OrganicMetal.data_loader(file_path=config['input_organic'],
                                                                   columns=['smiles', 'E12', 'Metal', 'Reaction'],
                                                                   reaction_type=config['reaction'],
                                                                   tensorize_fn=tensorize_with_subgraphs,
                                                                   batch_size=config['batch_size'],
                                                                   test_size=config['test_size'],)

metal_train_data, _ = OrganicMetal.data_loader(file_path='/work/u7069586/OMGNN/data/redox_metal.csv',
                                               columns=['Metal', 'Potential','Reaction'],
                                               reaction_type=config['reaction'], 
                                               tensorize_fn=metal_features,
                                               batch_size=config['batch_size'],
                                               is_metal=True,)
# organic_metal_train = organic_train_data + metal_train_data
organic_metal_train = organic_train_data 
organic_metal_train_loader = DataLoader(organic_metal_train, batch_size=config['batch_size'], shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OrganicMetal_GNN(node_dim=config['num_features'], 
                         bond_dim=11, 
                         hidden_dim=config['num_features'], 
                         output_dim=config['output_size'], 
                         dropout=config['dropout']).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['anneal_rate'])
criterion_cla = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

"""
3. 開始訓練
"""
train_loss_history, train_reg_history, train_cla_history, test_loss_history, test_reg_history, test_cla_history, test_accuracy_history = [], [], [], [], [], [], []
for epoch in range(config['num_epochs']):
    model.train()
    total_loss, total_cla_loss, total_reg_loss, count = 0, 0, 0, 0
    for i,batch in enumerate(organic_metal_train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)

        potential_clas, potential_regs, losses = model(batch)

        all_loss, loss_cla, loss_reg = losses[0], losses[1], losses[2]
        total_loss += all_loss.item()
        total_cla_loss += loss_cla.item()
        total_reg_loss += loss_reg.item()
        count += 1
        optimizer.step()

    train_loss, train_reg_loss, train_cla_loss = (total_loss / count), (total_reg_loss / count), (total_cla_loss / count)
    test_loss, test_reg_loss, test_cla_loss, test_accuracy  = OrganicMetal.evaluate_model(model, 
                                                                                          orgainc_test_loader, 
                                                                                          device, 
                                                                                          output_file=None)
    print(f"Epoch {epoch}, Train RMSE Loss: {train_loss:.4f}")
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    train_reg_history.append(train_reg_loss)
    train_cla_history.append(train_cla_loss)
    test_reg_history.append(test_reg_loss)
    test_cla_history.append(test_cla_loss)
    test_accuracy_history.append(test_accuracy)
    df_loss = pd.DataFrame({'train_loss': train_loss_history, 
                            'test_loss': test_loss_history, 
                            'train_reg_loss': train_reg_history, 
                            'test_reg_loss': test_reg_history, 
                            'train_cla_loss': train_cla_history, 
                            'test_cla_loss': test_cla_history, 
                            'test_cla_accuray':test_accuracy_history})
    df_loss.to_csv(os.path.join(config['save_path'], f"{config['version']}-{config['batch_size']}-{config['num_epochs']}_loss.csv"))



"""
4. 評估模型
"""
OrganicMetal.evaluate_model(model, organic_metal_train_loader, device, 
                            output_file=f"{config['version']}-{config['reaction']}-{config['mode']}-{config['batch_size']}-{config['num_epochs']}_train_pred_true.csv")
OrganicMetal.evaluate_model(model, orgainc_test_loader, device, output_file=f"{config['version']}-{config['reaction']}-{config['mode']}-{config['batch_size']}-{config['num_epochs']}_valid_pred_true.csv")

torch.save(model.state_dict(), os.path.join(config['model_path'], f"{config['version']}-{config['batch_size']}-{config['num_epochs']}.pkl"))