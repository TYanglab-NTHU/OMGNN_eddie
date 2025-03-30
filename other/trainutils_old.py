import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch    import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch_geometric.data     import Data, DataLoader

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from chemutils import *
from optparse  import OptionParser

def load_config_by_version(csv_path, version):
    """
    Loads configuration parameters from a CSV file for a specific version.
    
    The CSV should have a header row with parameter names, including a "version" column.
    """
    df = pd.read_csv(csv_path)
    # DataFrame for the given version.
    config_row = df[df['version'] == version]
    if config_row.empty:
        raise ValueError(f"No configuration found for version {version}")
    config = config_row.iloc[0].to_dict()
    
    # Convert parameters to appropriate types.
    # Adjust the casting based on your parameter types.
    config['test_size']    = float(config['test_size'])
    config['num_features'] = int(config['num_features'])
    config['output_size']  = int(config['output_size'])
    config['batch_size']   = int(config['batch_size'])
    config['num_epochs']   = int(config['num_epochs'])
    config['dropout']      = float(config['dropout'])
    
    return config


class OrganicMetal():
    def __init__(self):
        pass

    def data_loader(file_path, columns, tensorize_fn, batch_size, reaction_type='reduction', test_size=0.2, is_metal=False):
        df = pd.read_csv(file_path)
        df = df[columns] 

        if not is_metal:
            df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))))
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        else: # All metal data is used for training
            df['Potential'] = df['Potential'] - 0.63
            train_data, _ = df, None

        def tensorize_dataset(data, is_metal=False):
            dataset = []
            for _, row in data.iterrows():
                try:
                    if is_metal:
                        fatoms, graphs, edge_features = tensorize_fn(row["Metal"])
                        fatoms = torch.unsqueeze(fatoms, dim=0)
                        label = torch.Tensor([row["Potential"]])
                        name = [row["Metal"]]
                        data_item = Data(
                        x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
                    else:
                        [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], row["Metal"])
                        label = torch.Tensor(row["E12"])
                        name  = fatoms[1]

                        data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
                        data_item.midx = midx
                    dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
        if is_metal:
            return train_dataset, None
        test_dataset = tensorize_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_loader
    
    def evaluate_model(model, loader, device, output_file=""):
        model.eval()
        names = []
        eval_actuals_reg, eval_predictions_reg = [], []
        eval_actuals_cla, eval_predictions_cla = [], []
        total_loss, total_cla_loss, total_reg_loss, count, total_batches, correct_batches = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for data in loader:
                try:
                    actuals, predictions = "" , ""
                    data = data.to(device)
                    potential_clas, potential_regs, losses = model.sample(data)

                    for i, real in enumerate(data.ys):
                        actuals     += str(real.cpu().numpy()) + ","
                        predictions += str(potential_regs[i].squeeze().cpu().detach().numpy()) + ","

                    redox_num_tensor = potential_clas[0]
                    num_peaks        = torch.argmax(redox_num_tensor, dim=0).item()
                    actuals_cla     = "".join(str(len(data.ys)))
                    predictions_cla = "".join(str(num_peaks))

                    if num_peaks == len(data.ys):
                        correct_batches += 1
                    total_batches += 1

                    eval_actuals_reg.append(actuals.strip(','))
                    eval_predictions_reg.append(predictions.strip(','))

                    eval_actuals_cla.append(actuals_cla)
                    eval_predictions_cla.append(predictions_cla)

                    names.append(data.name)
                    all_loss, loss_cla, loss_reg = losses[0], losses[1], losses[2]
                    total_loss += all_loss.item()
                    total_cla_loss += loss_cla.item()
                    total_reg_loss += loss_reg.item()
                    count += 1

                except Exception as e:
                    print(f"Error evaluating model: {e}")

        df_reg = pd.DataFrame({
        "Actuals"    : eval_actuals_reg,
        "Predictions": eval_predictions_reg,
        "SMILES"     : names
        })
        df_cla = pd.DataFrame({
            "Actuals"    : eval_actuals_cla,
            "Predictions": eval_predictions_cla,
            "SMILES"     : names
        })
        df_reg.to_csv(os.path.join("../results/", f"reg-{output_file}"), index=False)
        df_cla.to_csv(os.path.join("../results/", f"cla-{output_file}"), index=False)
        return total_loss / count, total_reg_loss / count, total_cla_loss / count, correct_batches / total_batches if total_batches > 0 else 0.0

class Solvent_OrganicMetal():
    def __init__(self):
        pass

    def data_loader(file_path, columns, tensorize_fn, batch_size, reaction_type='reduction', test_size=0.2, is_metal=False):
        df = pd.read_csv(file_path)
        df = df[columns] 

        if not is_metal:
            df['IE / eV']  = df['IE / eV'].apply(lambda x: list(map(float, x.split(','))))
            df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))))
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        def tensorize_dataset(data, is_metal=False):
            dataset = []
            for _, row in data.iterrows():
                try:
                    if is_metal:
                        fatoms, graphs, edge_features = tensorize_fn(row["Metal"])
                        fatoms = torch.unsqueeze(fatoms, dim=0)
                        label = torch.Tensor([row["Potential"]])
                        name = [row["Metal"]]
                        data_item = Data(
                        x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
                    else:
                        [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], 'None')
                        label   = torch.Tensor(row['IE / eV'])
                        label2  = torch.Tensor(row['E12'])
                        name    = fatoms[1]
                        solvent = row['Solvent']
                        data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=label, ys2=label2, solvent=solvent, name=name, reaction=row["Reaction"])   #reaction=row["Reaction"])
                        data_item.midx = midx
                    dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
        if is_metal:
            return train_dataset, None
        test_dataset = tensorize_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_loader
    
    def evaluate_model(model, loader, device, output_file=""):
        model.eval()
        names = []
        eval_actuals_reg, eval_predictions_reg = [], []
        eval_actuals_del, eval_predictions_del = [], []
        eval_actuals_cla, eval_predictions_cla = [], []
        total_loss, total_cla_loss, total_reg_loss, total_delta_loss, count, total_batches, correct_batches = 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for data in loader:
                try:
                    actuals, predictions, delta_actuals, delta_preds = "" , "", "", ""
                    data = data.to(device)
                    potential_clas, potential_regs, potential_delta, losses = model.sample(data)

                    for i, real in enumerate(data.ys):
                        actuals     += str(real.squeeze().cpu().detach().numpy()) + ","
                        delta_actuals += str(data.ys[i].squeeze().cpu().detach().numpy() - data.ys2[i].squeeze().cpu().detach().numpy()) + ","
                        predictions += str(potential_regs[i].squeeze().cpu().detach().numpy()) + ","
                        delta_preds += str(potential_delta[i].squeeze().cpu().detach().numpy()) + ","
                        

                    redox_num_tensor = potential_clas[0]
                    num_peaks        = torch.argmax(redox_num_tensor, dim=0).item()
                    actuals_cla     = "".join(str(len(data.ys)))
                    predictions_cla = "".join(str(num_peaks))

                    if num_peaks == len(data.ys):
                        correct_batches += 1
                    total_batches += 1

                    eval_actuals_reg.append(actuals.strip(','))
                    eval_predictions_reg.append(predictions.strip(','))
                    eval_actuals_del.append(delta_actuals.strip(','))
                    eval_predictions_del.append(delta_preds.strip(','))

                    eval_actuals_cla.append(actuals_cla)
                    eval_predictions_cla.append(predictions_cla)

                    names.append(data.name)
                    all_loss, loss_cla, loss_reg = losses[0], losses[1], losses[2]
                    total_loss += all_loss.item()
                    total_cla_loss += loss_cla.item()
                    total_reg_loss += loss_reg.item()
                    count += 1

                except Exception as e:
                    print(f"Error evaluating model: {e}")

        df_reg = pd.DataFrame({
        "Actuals"     : eval_actuals_reg,
        "Predictions" : eval_predictions_reg,
        "Actual_delta": eval_actuals_del,
        "Pred_delta"  : eval_predictions_del,
        "SMILES"      : names
        })
        df_cla = pd.DataFrame({
            "Actuals"    : eval_actuals_cla,
            "Predictions": eval_predictions_cla,
            "SMILES"     : names
        })
        df_reg.to_csv(os.path.join("../results/", f"reg-{output_file}"), index=False)
        df_cla.to_csv(os.path.join("../results/", f"cla-{output_file}"), index=False)
        return total_loss / count, total_reg_loss / count, total_cla_loss / count, correct_batches / total_batches if total_batches > 0 else 0.0    

    # def data_loader(file_path, columns, tensorize_fn, batch_size, reaction_type='reduction', test_size=0.2, is_metal=False):
    #     df = pd.read_csv(file_path)
    #     df = df[columns] 

    #     if not is_metal:
    #         df['IE / eV']  = df['IE / eV']   #.apply(lambda x: list(map(float, x.split(','))))
            # df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))))
    #         train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    #         all_values = [value for value in train_data['IE / eV']]

    #         global_mean = np.mean(all_values)
    #         global_std = np.std(all_values)
    #         df = pd.DataFrame({'mean': [global_mean], 'std': [global_std]})
    #         df.to_csv('../results/IE_zscore.csv', index=False)
    #         def compute_z_scores(x):
    #             return [(x - global_mean) / global_std]

    #         train_data['IE_zscore'] = train_data['IE / eV'].apply(compute_z_scores)
    #         test_data['IE_zscore']  = test_data['IE / eV'].apply(compute_z_scores)

    #     else: # All metal data is used for training
    #         df['Potential'] = df['Potential'] - 0.63
    #         train_data, _ = df, None

    #     def tensorize_dataset(data, is_metal=False):
    #         dataset = []
    #         for _, row in data.iterrows():
    #             try:
    #                 if is_metal:
    #                     fatoms, graphs, edge_features = tensorize_fn(row["Metal"])
    #                     fatoms = torch.unsqueeze(fatoms, dim=0)
    #                     label = torch.Tensor([row["Potential"]])
    #                     name = [row["Metal"]]
    #                     data_item = Data(
    #                     x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
    #                 else:
    #                     [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], 'None')
    #                     # label = torch.Tensor(row['IE / eV'])
    #                     # label = row['IE_zscore']
    #                     label = row['IE / eV']
    #                     name  = fatoms[1]

    #                     data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=label, name=name)   #reaction=row["Reaction"])
    #                     data_item.midx = midx
    #                 dataset.append(data_item)
    #             except Exception as e:
    #                 print(f"Error processing row: {e}")
    #                 continue
    #         return dataset

    #     train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
    #     if is_metal:
    #         return train_dataset, None
    #     test_dataset = tensorize_dataset(test_data)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #     return train_dataset, test_loader
    
    # def evaluate_model(model, loader, device, output_file=""):
    #     model.eval()
    #     names = []
    #     eval_actuals_reg, eval_predictions_reg = [], []
    #     eval_actuals_cla, eval_predictions_cla = [], []
    #     total_loss, total_cla_loss, total_reg_loss, count, total_batches, correct_batches = 0, 0, 0, 0, 0, 0
    #     with torch.no_grad():
    #         for data in loader:
    #             try:
    #                 actuals, predictions = "" , ""
    #                 data = data.to(device)
    #                 potential_clas, potential_regs, losses = model.sample(data)

    #                 for i, real in enumerate(data.ys):
    #                     actuals     += str(real.squeeze().cpu().detach().numpy()) + ","
    #                     predictions += str(potential_regs[i].squeeze().cpu().detach().numpy()) + ","

    #                 redox_num_tensor = potential_clas[0]
    #                 num_peaks        = torch.argmax(redox_num_tensor, dim=0).item()
    #                 actuals_cla     = "".join(str(len(data.ys)))
    #                 predictions_cla = "".join(str(num_peaks))

    #                 if num_peaks == len(data.ys):
    #                     correct_batches += 1
    #                 total_batches += 1

    #                 eval_actuals_reg.append(actuals.strip(','))
    #                 eval_predictions_reg.append(predictions.strip(','))

    #                 eval_actuals_cla.append(actuals_cla)
    #                 eval_predictions_cla.append(predictions_cla)

    #                 names.append(data.name)
    #                 all_loss, loss_cla, loss_reg = losses[0], losses[1], losses[2]
    #                 total_loss += all_loss.item()
    #                 total_cla_loss += loss_cla.item()
    #                 total_reg_loss += loss_reg
    #                 count += 1

    #             except Exception as e:
    #                 print(f"Error evaluating model: {e}")

    #     df_reg = pd.DataFrame({
    #     "Actuals"    : eval_actuals_reg,
    #     "Predictions": eval_predictions_reg,
    #     "SMILES"     : names
    #     })
    #     df_cla = pd.DataFrame({
    #         "Actuals"    : eval_actuals_cla,
    #         "Predictions": eval_predictions_cla,
    #         "SMILES"     : names
    #     })
    #     df_reg.to_csv(os.path.join("../results/", f"reg-{output_file}"), index=False)
    #     df_cla.to_csv(os.path.join("../results/", f"cla-{output_file}"), index=False)
    #     return total_loss / count, total_reg_loss / count, total_cla_loss / count, correct_batches / total_batches if total_batches > 0 else 0.0    

class OM():
    def __init__(self):
        pass

    def data_loader_z(file_path, columns, tensorize_fn, batch_size, reaction_type='reduction', test_size=0.2, is_metal=False, is_redox=False, is_split_redox=True):
        df = pd.read_csv(file_path)
        df = df[columns] 

        if is_split_redox:
            df = df[df['Reaction'] == reaction_type]

        if not is_metal:
            df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))))
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
            # z_score
            all_values = [value for sublist in train_data['E12'] for value in sublist]

            global_mean = np.mean(all_values)
            global_std = np.std(all_values)
            def compute_z_scores(values):
                return [(x - global_mean) / global_std for x in values]

            train_data['E12_zscore'] = train_data['E12'].apply(compute_z_scores)
            test_data['E12_zscore']  = test_data['E12'].apply(compute_z_scores)

        else: # All metal data is used for training
            df['Potential'] = df['Potential'] - 0.63
            train_data, _   = df, None

        def tensorize_dataset(data, is_metal=False):
            dataset = []
            for _, row in data.iterrows():
                try:
                    if is_metal:
                        fatoms, graphs, edge_features = tensorize_fn(row["Metal"])
                        fatoms = torch.unsqueeze(fatoms, dim=0)
                        label  = torch.Tensor([row["Potential"]])
                        name   = [row["Metal"]]
                        data_item = Data(
                        x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
                    else:
                        [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], row["Metal"])
                        label = torch.Tensor(row['E12_zscore'])
                        name = fatoms[1]
                        if is_redox:
                            redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                            data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"], oreder_site=row["redox_site_smiles"])
                            data_item.midx = midx
                        else:
                            data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
                            data_item.midx = midx
                    dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
        if is_metal:
            return train_dataset, None
        test_dataset = tensorize_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_loader, global_mean, global_std


    def evaluate_model(model, loader, device, output_file=""):
        model.eval()
        names = []
        eval_actuals_reg, eval_predictions_reg = [], []
        eval_actuals_cla, eval_predictions_cla = [], []
        total_loss, total_cla_loss, total_reg_loss, count, total_batches, correct_batches = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for data in loader:
                try:
                    actuals, predictions = "" , ""
                    loss_cla, loss_reg = 0, 0
                    data = data.to(device)
                    num_logit, num_peak, E12_regs = model.sample(data, device)

                    for i, real in enumerate(data.ys):
                        actuals     += str(real.cpu().numpy()) + ","
                        if i < len(E12_regs):
                            predictions += str(E12_regs[i].squeeze().cpu().detach().numpy()) + ","
                            loss_reg    += F.mse_loss(E12_regs[i].squeeze(), real).item() 
                        else:
                            break

                    real_num_redox = [data.redox[i][0][1] for i in range(len(data.redox))]
                    actuals_cla     = ",".join(map(str, real_num_redox))
                    predictions_cla = ",".join(map(str, num_peak.cpu().tolist()))
                    loss_cla       = F.cross_entropy(num_logit, torch.tensor(real_num_redox).to(device))


                    for j, num in enumerate(num_peak):
                        total_batches += 1
                        if num == real_num_redox[j]:
                            correct_batches += 1    
                        
                    eval_actuals_reg.append(actuals.strip(','))
                    eval_predictions_reg.append(predictions.strip(','))

                    eval_actuals_cla.append(actuals_cla)
                    eval_predictions_cla.append(predictions_cla)

                    names.append(data.name)
                    all_loss = loss_cla + loss_reg
                    total_loss += all_loss.item()
                    total_cla_loss += loss_cla.item()
                    total_reg_loss += loss_reg
                    count += 1

                except Exception as e:
                    print(f"Error evaluating model: {e}")

        df_reg = pd.DataFrame({
        "Actuals"    : eval_actuals_reg,
        "Predictions": eval_predictions_reg,
        "SMILES"     : names
        })
        df_cla = pd.DataFrame({
            "Actuals"    : eval_actuals_cla,
            "Predictions": eval_predictions_cla,
            "SMILES"     : names
        })
        df_reg.to_csv(os.path.join("../results/", f"reg-{output_file}"), index=False)
        df_cla.to_csv(os.path.join("../results/", f"cla-{output_file}"), index=False)
        return total_loss / count, total_reg_loss / count, total_cla_loss / count, correct_batches / total_batches if total_batches > 0 else 0.0


class OMGNN_v2():
    def __init__(self):
        pass


    def data_loader_z(file_path, columns, tensorize_fn, batch_size, reaction_type='reduction', test_size=0.2, is_metal=False, is_redox=False, is_split_redox=True):
        df = pd.read_csv(file_path)
        df = df[columns] 

        if is_split_redox:
            df = df[df['Reaction'] == reaction_type]

        if not is_metal:
            df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))))
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
            # z_score
            all_values = [value for sublist in train_data['E12'] for value in sublist]

            global_mean = np.mean(all_values)
            global_std = np.std(all_values)
            def compute_z_scores(values):
                return [(x - global_mean) / global_std for x in values]

            train_data['E12_zscore'] = train_data['E12'].apply(compute_z_scores)
            test_data['E12_zscore']  = test_data['E12'].apply(compute_z_scores)

        else: # All metal data is used for training
            df['Potential'] = df['Potential'] - 0.63
            train_data, _   = df, None
        
        def tensorize_dataset(data, is_metal=False):
            dataset = []
            for _, row in data.iterrows():
                try:
                    if is_metal:
                        fatoms, graphs, edge_features = tensorize_fn(row["Metal"])
                        fatoms = torch.unsqueeze(fatoms, dim=0)
                        label  = torch.Tensor([row["Potential"]])
                        name   = [row["Metal"]]
                        data_item = Data(
                        x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
                    else:
                        [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], row["Metal"])
                        label = torch.Tensor(row['E12_zscore'])
                        name = fatoms[1]
                        # if is_redox:
                        #     redox_idxs = redox_idx_smiles_modified([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                        #     data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"])
                        #     data_item.midx = midx
                        if is_redox:
                            redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                            data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"], oreder_site=row["redox_site_smiles"])
                            data_item.midx = midx
                        else:
                            data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
                            data_item.midx = midx
                    dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
        if is_metal:
            return train_dataset, None
        test_dataset = tensorize_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_loader, global_mean, global_std


    def evaluate_model_v2(model, loader, device, output_path="../results/", output_file="evaluation_results.csv"):
        model.eval()
        names, reactions = [], []
        eval_actuals_reg, eval_predictions_reg = [], []
        eval_actuals_cla, eval_predictions_cla = [], []
        total_loss, reg_loss, cla_loss, count = 0, 0, 0, 0
        with torch.no_grad():
            for data in loader:
                try:
                    data = data.to(device)
                    E12s, redox_num, _ = model(data, device)
                    
                    redox_order = data.oreder_site[0].split('/')
                    index_order = []
                    for item in redox_order:
                        matching_keys = [key for key, values in data.redox.items() if any(value[0] == item for value in values)]
                        if len(matching_keys) == 1:
                            index_order.append(matching_keys[0])
                        else:
                            index_order.append(matching_keys)
                    
                    real_redox_num = []
                    for i in range(len(data.redox)):
                        real_redox_num.append(data.redox[i][0][1])
                    target = torch.tensor(real_redox_num, device=device)
                    
                    loss_cla, loss_reg = 0, 0
                    actuals_reg_str, predictions_reg_str = "", ""
                    
                    for j, real in enumerate(data.ys):
                        if j >= len(E12s):
                            break
                        
                        redox_num_tensor = torch.stack(redox_num[j], dim=0)
                        num_peaks        = torch.argmax(redox_num_tensor, dim=1)
                        if j == 0:
                            actuals_cla     = ",".join(map(str, target.cpu().tolist()))
                            predictions_cla = ",".join(map(str, num_peaks.cpu().tolist()))
                        loss_cla =  F.cross_entropy(redox_num_tensor, target)
                        loss_reg += F.mse_loss(E12s[j], real)
                        
                        if j < len(index_order):
                            indices = index_order[j]
                            new_target = target.clone()
                            if isinstance(indices, list):
                                for k in indices:
                                    if new_target[k] > 0:
                                        new_target[k]   -= 1
                            else:
                                if new_target[indices] > 0:
                                    new_target[indices] -= 1
                            target = new_target
                        
                        actuals_reg_str += str(real.cpu().detach().item()) + ","
                        predictions_reg_str += str(E12s[j].cpu().detach().item()) + ","
                    
                    loss = loss_cla / 5 + loss_reg
                    count += 1
                    total_loss += loss.item()
                    reg_loss   += loss_reg.item()
                    cla_loss   += loss_cla.item()
                    
                    eval_actuals_reg.append(actuals_reg_str.strip(','))
                    eval_predictions_reg.append(predictions_reg_str.strip(','))
                    
                    eval_actuals_cla.append(actuals_cla)
                    eval_predictions_cla.append(predictions_cla)

                    names.extend(data.name[0])
                    reactions.extend(data.reaction)
                    
                except Exception as e:
                    print(f"An error occurred: {e}")
        
        # Save regression and classification evaluation results to CSV files
        df_reg = pd.DataFrame({
            "Actuals": eval_actuals_reg,
            "Predictions": eval_predictions_reg,
            "SMILES": names,
            "reaction": reactions
        })
        df_cla = pd.DataFrame({
            "Actuals": eval_actuals_cla,
            "Predictions": eval_predictions_cla,
            "SMILES": names,
            "reaction": reactions
        })
        df_reg.to_csv(os.path.join(output_path, f"reg-{output_file}.csv"), index=False)
        df_cla.to_csv(os.path.join(output_path, f"cla-{output_file}.csv"), index=False)
        
        return total_loss / count, reg_loss / count, cla_loss / count