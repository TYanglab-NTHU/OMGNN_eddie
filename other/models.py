import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing, GCNConv,  Linear, BatchNorm, GlobalAttention, GATConv

class BondMessagePassing(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=5, dropout=0.3):
        super(BondMessagePassing, self).__init__()
        self.W_i  = nn.Linear(node_features + bond_features, hidden_size)
        self.W_h  = nn.Linear(hidden_size, hidden_size)
        self.W_o  = nn.Linear(node_features + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.depth = depth 

    def update(self, M_t, H_0):
        H_t = self.W_h(M_t)
        H_t = self.relu(H_0 + H_t)
        H_t = self.dropout(H_t)

        return H_t

    def message(self, H, batch):
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)[batch.edge_index[0]]
        M_rev = H[batch.rev_edge_index]

        # degree = torch.bincount(batch.edge_index[1], minlength=len(batch.edge_index[1])).unsqueeze(1).to(H.device)
        # degree = torch.where(degree == 0, torch.ones_like(degree), degree)
        # M_all = M_all / degree
        return M_all - M_rev

    def forward(self, batch):
        H_0 = self.W_i(torch.cat([batch.x[batch.edge_index[0]], batch.edge_attr], dim=1))
        H   = self.relu(H_0)
        for _ in range(1, self.depth):
            M = self.message(H, batch)
            H = self.update(M, H_0)
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1]) # Noramal GNN tran
        M = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
        M = torch.where(M.sum(dim=1, keepdim=True) == 0, batch.x, M)            
        H = self.W_o(torch.cat([batch.x, M], dim=1))
        H = self.relu(H)    
        H = self.dropout(H)
        return H             

class OrganicMetal_GNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, dropout=0.1):
        super(OrganicMetal_GNN, self).__init__()
        self.GCN1  = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=dropout)
        self.pool  = global_mean_pool
        
        # 分類任務
        self.potential_cla = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))  # [0, 1, 2, 3, 4]
        # 回歸任務
        self.potential_reg = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))

        # 氧化的回歸器
        self.potential_reg_ox = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))

        # 還原的回歸器
        self.potential_reg_red = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        # 損失函數
        self.criterion_cla = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        # 門控RNN
        self.rnn_gate      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())
        
    # 獲取反向邊緣索引
    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i 
                         for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index

    # 子圖的前向傳播
    def forward_subgraph1(self, x, subgraph1_edge_index, batch1, edge_attr):
        # 獲取反向邊緣索引
        subgraph1_rev_edge_index = self._rev_edge_index(subgraph1_edge_index)
        # 構建子圖
        subgraph1_batch = Data(x=x,
                               edge_index=subgraph1_edge_index,
                               rev_edge_index=subgraph1_rev_edge_index,
                               edge_attr=edge_attr)
        # 進行前向傳播
        subgraph1_result = self.GCN1(subgraph1_batch)
        # 進行池化
        subgraph1_result_ = self.pool(subgraph1_result, batch1)
        return subgraph1_result, subgraph1_result_

    # 主函數
    def forward(self, batch):
        # 主前向傳播方法，處理整個批次的數據
        # 從批次中提取單個圖的數據（注意：這裡沒有處理多個圖的情況，只處理了最後一個圖）
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, true_potentials, reaction = graph.x, graph.edge_index, graph.edge_attr, graph.ys, graph.reaction
        # 從edge_index中提取子圖的邊索引和批次信息
        subgraph1_edge_index, batch1 = edge_index
        # 處理子圖，只保留圖級特徵
        _, subgraph1_result_ = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr) 

        # 創建張量表示氧化還原峰的數量，初始值為真實電位的數量
        redox_num     = torch.tensor([len(true_potentials)], device='cuda')
        potential_clas, potential_regs = torch.tensor([], device='cuda'), torch.tensor([], device='cuda')
        losses, loss_clas, loss_regs = 0, 0, 0
        # 遍歷真實電位列表，進行分類和回歸任務
        for i, true_potential in enumerate(true_potentials):
            # 進行分類任務
            potential_cla = self.potential_cla(subgraph1_result_)
            # 根據反應類型選擇回歸任務
            if reaction == 'reduction':
                potential_reg = self.potential_reg_red(subgraph1_result_)
            else:
                potential_reg = self.potential_reg_ox(subgraph1_result_)
            # 計算分類損失
            loss_cla = self.criterion_cla(potential_cla, redox_num)
            # 計算回歸損失
            loss_reg = self.criterion_reg(potential_reg.squeeze(), true_potential)
            # 總損失
            loss     = loss_cla + loss_reg
            # 累加損失
            loss_clas += loss_cla
            loss_regs += loss_reg
            losses    += loss
            # 反向傳播
            loss.backward(retain_graph=True)
            # 累加分類和回歸結果
            potential_clas = torch.cat((potential_clas, potential_cla), 0)
            potential_regs = torch.cat((potential_regs, potential_reg), 0)

            redox_num  = redox_num - 1
            redox_node = _.clone()
            # 使用門控機制更新節點特徵
            redox_node = redox_node * self.rnn_gate(redox_node) + redox_node

            _, subgraph1_result_ = self.forward_subgraph1(redox_node, subgraph1_edge_index, batch1, edge_attr) 

        # "last cla lig no redox peak"
        potential_cla = self.potential_cla(subgraph1_result_)
        loss_cla   = self.criterion_cla(potential_cla, redox_num)
        loss       = loss_cla
        loss_clas += loss_cla
        losses    += loss
        loss.backward(retain_graph=True)
        potential_clas = torch.cat((potential_clas, potential_cla), 0)

        return potential_clas, potential_regs, (losses, loss_clas, loss_regs)

    def sample(self, batch):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, true_potentials, reaction = graph.x, graph.edge_index, graph.edge_attr, graph.ys, graph.reaction
        subgraph1_edge_index, batch1 = edge_index

        _, subgraph1_result_ = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr) 

        redox_num = torch.tensor([len(true_potentials)], device='cuda')
        potential_clas, potential_regs = torch.tensor([], device='cuda'), torch.tensor([], device='cuda')
        losses, loss_clas, loss_regs = 0, 0, 0
        for i, true_potential in enumerate(true_potentials):
            potential_cla = self.potential_cla(subgraph1_result_)
            if reaction == 'reduction':
                potential_reg = self.potential_reg_red(subgraph1_result_)
            else:
                potential_reg = self.potential_reg_ox(subgraph1_result_)
            loss_cla = self.criterion_cla(potential_cla, redox_num)
            loss_reg = self.criterion_reg(potential_reg.squeeze(), true_potential)
            loss     = loss_cla + loss_reg
            loss_clas += loss_clas + loss_cla
            loss_regs += loss_regs + loss_reg
            losses    += loss
            potential_clas = torch.cat((potential_clas, potential_cla), 0)
            potential_regs = torch.cat((potential_regs, potential_reg), 0)

            # "update GCN1 features"
            redox_num  = redox_num - 1
            redox_node = _.clone()
            redox_node = redox_node * self.rnn_gate(redox_node) + redox_node

            _, subgraph1_result_ = self.forward_subgraph1(redox_node, subgraph1_edge_index, batch1, edge_attr) 

        # "last cla lig no redox peak"
        potential_cla  = self.potential_cla(subgraph1_result_)
        loss_cla       = self.criterion_cla(potential_cla, redox_num)
        loss           = loss_cla
        loss_clas     += loss_cla
        losses        += loss
        potential_clas = torch.cat((potential_clas, potential_cla), 0)

        return potential_clas, potential_regs, (losses, loss_clas, loss_regs)

solvent_dict = {'ACN': [3.92*0.1 , 36.6*0.01]}







































class Solvent_OrganicMetal_GNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, dropout=0.1):
        super(OrganicMetal_GNN, self).__init__()
        self.GCN1  = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=dropout)
        self.pool  = global_mean_pool

        self.potential_cla = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))  # [0, 1, 2, 3, 4]
        
        self.IP_reg = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        
        self.delta_potentail = nn.Sequential(
            nn.Linear(hidden_dim + 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        
        self.criterion_cla = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        self.rnn_gate      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())

    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i 
                         for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index

    def forward_subgraph1(self, x, subgraph1_edge_index, batch1, edge_attr):
        subgraph1_rev_edge_index = self._rev_edge_index(subgraph1_edge_index)
        subgraph1_batch = Data(x=x,
                               edge_index=subgraph1_edge_index,
                               rev_edge_index=subgraph1_rev_edge_index,
                               edge_attr=edge_attr)
        subgraph1_result  = self.GCN1(subgraph1_batch)
        subgraph1_result_ = self.pool(subgraph1_result, batch1)
        return subgraph1_result, subgraph1_result_

    def forward(self, batch):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, true_IPs, true_potentials, solvent, reaction = graph.x, graph.edge_index, graph.edge_attr, graph.ys2, graph.ys, graph.solvent, graph.reaction
        subgraph1_edge_index, batch1 = edge_index

        _, subgraph1_result_ = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr) 

        solvent_features = solvent_dict[solvent]
        solvent_features = torch.Tensor(solvent_features).cuda().unsqueeze(0)

        redox_num     = torch.tensor([len(true_potentials)], device='cuda')
        potential_clas, potential_regs = torch.tensor([], device='cuda'), torch.tensor([], device='cuda')
        losses, loss_clas, loss_regs = 0, 0, 0
        for i, true_IP in enumerate(true_IPs):
            potential_cla = self.potential_cla(subgraph1_result_)
            IP_reg        = self.IP_reg(subgraph1_result_)
            delta_poten   = self.delta_potentail(torch.cat([subgraph1_result_, solvent_features], dim=1))
            true_delta    = true_IP - true_potentials[i]

            loss_cla   = self.criterion_cla(potential_cla, redox_num)
            loss_reg   = self.criterion_reg(IP_reg.squeeze(), true_IP)
            loss_delta = self.criterion_reg(delta_poten.squeeze(), true_delta)
            loss       = loss_cla + loss_reg + loss_delta
            loss_clas += loss_clas + loss_cla
            loss_regs += loss_regs + loss_reg
            losses    += loss
            loss.backward(retain_graph=True)
            potential_clas = torch.cat((potential_clas, potential_cla), 0)
            potential_regs = torch.cat((potential_regs, IP_reg), 0)

            "update GCN1 features"
            redox_num  = redox_num - 1
            redox_node = _.clone()
            redox_node = redox_node * self.rnn_gate(redox_node) + redox_node

            _, subgraph1_result_ = self.forward_subgraph1(redox_node, subgraph1_edge_index, batch1, edge_attr) 

        "last cla lig no redox peak"
        potential_cla = self.potential_cla(subgraph1_result_)
        loss_cla   = self.criterion_cla(potential_cla, redox_num)
        loss       = loss_cla
        loss_clas += loss_cla
        losses    += loss
        loss.backward(retain_graph=True)
        potential_clas = torch.cat((potential_clas, potential_cla), 0)

        return potential_clas, potential_regs, (losses, loss_clas, loss_regs)

    # def sample(self, batch):
    #     for graph in batch.to_data_list():
    #         x, edge_index, edge_attr, true_potentials, lig_reaction = graph.x, graph.edge_index, graph.edge_attr, graph.ys, graph.reaction
    #     subgraph1_edge_index, batch1 = edge_index

    #     _, subgraph1_result_ = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr) 

    #     redox_num     = torch.tensor([len(true_potentials)], device='cuda')
    #     potential_clas, potential_regs = torch.tensor([], device='cuda'), torch.tensor([], device='cuda')
    #     losses, loss_clas, loss_regs = 0, 0, 0
    #     for i, true_potential in enumerate(true_potentials):
    #         potential_cla = self.potential_cla(subgraph1_result_)
    #         potential_reg = self.potential_reg(subgraph1_result_)
    #         loss_cla = self.criterion_cla(potential_cla, redox_num)
    #         loss_reg = self.criterion_reg(potential_reg.squeeze(), true_potential)
    #         loss     = loss_cla + loss_reg
    #         loss_clas += loss_clas + loss_cla
    #         loss_regs += loss_regs + loss_reg
    #         losses    += loss
    #         potential_clas = torch.cat((potential_clas, potential_cla), 0)
    #         potential_regs = torch.cat((potential_regs, potential_reg), 0)

    #         "update GCN1 features"
    #         redox_num  = redox_num - 1
    #         redox_node = _.clone()
    #         redox_node = redox_node * self.rnn_gate(redox_node) + redox_node

    #         _, subgraph1_result_ = self.forward_subgraph1(redox_node, subgraph1_edge_index, batch1, edge_attr) 

    #     "last cla lig no redox peak"
    #     potential_cla = self.potential_cla(subgraph1_result_)
    #     loss_cla   = self.criterion_cla(potential_cla, redox_num)
    #     loss       = loss_cla
    #     loss_clas += loss_cla
    #     losses    += loss
    #     potential_clas = torch.cat((potential_clas, potential_cla), 0)

    #     return potential_clas, potential_regs, (losses, loss_clas, loss_regs)


class OMGNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, dropout=0.3, mode='auto'):
        super(OMGNN, self).__init__()
        self.GCN1 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=0.3)
        self.GCN2 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=0.3)
        self.GCN3 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=0.3)
        self.pool = global_mean_pool
        self.potential_reg = nn.Sequential(
            nn.Linear(hidden_dim + 1, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.num_peaks_red = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5),
            nn.Softmax())
        self.num_peaks_ox = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5),
            nn.Softmax())
        self.E12_reg_red = nn.Sequential(
            nn.Linear(hidden_dim , 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.E12_reg_ox = nn.Sequential(
            nn.Linear(hidden_dim , 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.gate_GCN3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())

    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index


    def forward_subgraph(self, x, edge_index, batch, edge_attr, gcn, pre_proc=None, transform_edge_attr=None):
        if pre_proc is not None:
            x = pre_proc(x)

        rev_edge_index = self._rev_edge_index(edge_index)

        if transform_edge_attr is not None:
            edge_attr = transform_edge_attr(edge_attr)

        data = Data(x=x, edge_index=edge_index, rev_edge_index=rev_edge_index, edge_attr=edge_attr)

        if isinstance(gcn, GATConv):
            result = gcn(x, edge_index, edge_attr) 
        else:
            result = gcn(data) 

        result_pooled = self.pool(result, batch)
        return result, result_pooled

    # def forward(self, batch):
    #     """只對redox site的分類loss backward,
    #     但最後ligands沒有redox site後所有一起backward"""
    #     for graph in batch.to_data_list():
    #         x, edge_index, edge_attr, midx, real_E12s, reaction, redox, redox_order = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox, graph.oreder_site

    #     subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
    #     subgraph1_edge_index, batch1 = subgraph1
    #     subgraph2_edge_index, batch2 = subgraph2
    #     subgraph3_edge_index, batch3 = subgraph3

    #     #"results after GCN and result_ after global pooling"
    #     subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
    #     subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
    #     subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

    #     total_loss = 0
    #     # convert batch1 index to batch3 index
    #     m_batch1  = batch1[midx]
    #     new_batch = batch2[batch1_2.long()]

    #     mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
    #     ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device='cuda')

    #     redox_order = redox_order.split('/')
    #     index_order = []
    #     for item in redox_order:
    #         matching_keys = [key for key, values in batch.redox.items() if any(value[0] == item for value in values)]
    #         if len(matching_keys) == 1:
    #             index_order.append(matching_keys[0])
    #         else:
    #             index_order.append(matching_keys)     

    #     num_each_redox = [redox[i][1] for i in range(len(redox))]
    #     batch1_subgraph3_result = subgraph3_result[ordered_indices]
    #     if   reaction == 'reduction':
    #         num_redox_all = self.num_peaks_red(batch1_subgraph3_result)
    #     elif reaction == 'oxidation':
    #         num_redox_all = self.num_peaks_ox(batch1_subgraph3_result)

    #     loss_cla = nn.CrossEntropyLoss()(num_redox_all, torch.tensor(num_each_redox, device='cuda'))
    #     loss = loss_cla / len(redox)
    #     loss.backward(retain_graph=True)
    #     total_loss += loss
    #     # batch1_subgraph3_result = subgraph3_result[ordered_indices]
    #     for i, site_index in enumerate(index_order):
    #         batch1_subgraph3_result = subgraph3_result[ordered_indices]
    #         if type(site_index) is list:
    #             break
    #         if   reaction == 'reduction':
    #             num_redox_site = self.num_peaks_red(batch1_subgraph3_result[site_index])
    #             E12            = self.E12_reg_red(batch1_subgraph3_result[site_index])
    #         elif reaction == 'oxidation':
    #             num_redox_site = self.num_peaks_ox(batch1_subgraph3_result[site_index])
    #             E12            = self.E12_reg_ox(batch1_subgraph3_result[site_index])
    #         loss_cla = nn.CrossEntropyLoss()(num_redox_site, torch.tensor(num_each_redox[site_index], device='cuda'))
    #         loss_reg = nn.MSELoss()(E12.squeeze(), real_E12s[i])
    #         loss = loss_cla * 10 + loss_reg
    #         loss.backward(retain_graph=True)
    #         total_loss += loss
    #         num_each_redox[site_index] = num_each_redox[site_index] - 1

    #         all_indices = torch.arange(subgraph3_result.shape[0], device='cuda')
    #         redox_node  = batch1_subgraph3_result[site_index].clone()
    #         redox_node  = redox_node * self.gate_GCN3(redox_node) + redox_node
    #         batch3_redox_idx = mapping_dict.get(site_index)
    #         nonredox_subgraph3_result = subgraph3_result[all_indices != batch3_redox_idx]
    #         updated_subgraph3_result  = nonredox_subgraph3_result.clone()

    #         subgraph3_result_ = torch.cat([updated_subgraph3_result[:batch3_redox_idx], redox_node.unsqueeze(0), updated_subgraph3_result[batch3_redox_idx:]], dim=0)
    #         subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph3_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

    #     return total_loss 
    
    def forward(self, batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, real_E12s, reaction, redox = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox
        
        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2 = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        #"results after GCN and result_ after global pooling"
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

        # convert batch1 index to batch3 index
        m_batch1  = batch1[midx]
        new_batch = batch2[batch1_2.long()]

        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device)

        #  number of redox peaks
        each_num_redox  = []
        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        for ligs_result in batch1_subgraph3_result:
            if   reaction == 'reduction':
                each_num_redox.append(self.num_peaks_red(ligs_result))
            elif reaction == 'oxidation':
                each_num_redox.append(self.num_peaks_ox(ligs_result))
        
        num_each_redox = [redox[i][1] for i in range(len(redox))]
        loss_cla = nn.CrossEntropyLoss()(torch.stack(each_num_redox, dim=0), torch.tensor(num_each_redox, device=device))
        loss = loss_cla / len(redox)
        loss.backward(retain_graph=True)

        each_num_redox_ = torch.argmax(torch.stack(each_num_redox, dim=0), dim=1)
        redox_sites     = []
        for i, count in enumerate(each_num_redox_.tolist()):
            redox_sites.extend([i] * count)

        redox_idxs = []
        E12s = []
        if redox_sites != []:
            unique_redox_sites = list(set(redox_sites))
            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            if   reaction == 'reduction':
                lig_potentials = self.E12_reg_red(batch1_subgraph3_result[unique_redox_sites])
                E12, idx = lig_potentials.max(dim=0)
            elif reaction == 'oxidation':
                lig_potentials = self.E12_reg_ox(batch1_subgraph3_result[unique_redox_sites])
                E12, idx = lig_potentials.min(dim=0)
            E12s.append(E12)
            if real_E12s.shape[0] == 0:
                pass
            else:
                loss_reg = nn.MSELoss()(E12.squeeze(), real_E12s[0])
                loss_reg.backward(retain_graph=True)
                real_E12s = real_E12s[1:]

            redox_site_idx      = unique_redox_sites[idx]
            redox_idxs.append(redox_site_idx)
            redox_site_result   = batch1_subgraph3_result[redox_site_idx]
            redox_site_result_  = redox_site_result.unsqueeze(0)

            redox_pos = [redox[redox_site_idx][0]]
            sites = [i for i in range(len(redox)) if redox[i][0] in redox_pos]
            for site in sites:
                if site in redox_sites:
                    redox_sites.remove(site)

            all_indices = torch.arange(subgraph3_result.shape[0], device=device)

            "gate = sigmoid(FFN(x)) -1, 1"
            redox_site_change = redox_site_result_ * self.gate_GCN3(redox_site_result_) + redox_site_result_

            batch3_redox_idx  = mapping_dict.get(redox_site_idx)

            nonredox_subgraph3_result = subgraph3_result[all_indices != batch3_redox_idx]
            updated_subgraph3_result  = nonredox_subgraph3_result.clone()

            subgraph3_result_ = torch.cat([updated_subgraph3_result[:batch3_redox_idx], redox_site_change, updated_subgraph3_result[batch3_redox_idx:]], dim=0)

            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph3_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))


        while redox_sites:
            each_num_redox = []
            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            for ligs_result in batch1_subgraph3_result:
                if   reaction == 'reduction':
                    each_num_redox.append(self.num_peaks_red(ligs_result))
                elif reaction == 'oxidation':
                    each_num_redox.append(self.num_peaks_ox(ligs_result))

            loss_cla = nn.CrossEntropyLoss()(torch.stack(each_num_redox, dim=0), torch.tensor(num_each_redox, device=device))
            
            each_num_redox_    = torch.argmax(torch.stack(each_num_redox, dim=0), dim=1)
            unique_redox_sites = list(set(redox_sites))
            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            if   reaction == 'reduction':
                lig_potentials = self.E12_reg_red(batch1_subgraph3_result[unique_redox_sites])
                E12, idx = lig_potentials.max(dim=0)
            elif reaction == 'oxidation':
                lig_potentials = self.E12_reg_ox(batch1_subgraph3_result[unique_redox_sites])
                E12, idx = lig_potentials.min(dim=0)
            E12s.append(E12)
            if real_E12s.shape[0] == 0:
                pass
            else:
                loss_reg = nn.MSELoss()(E12.squeeze(), real_E12s[0])
                loss_reg.backward(retain_graph=True)
                real_E12s = real_E12s[1:]


            redox_site_idx       = unique_redox_sites[idx]
            redox_idxs.append(redox_site_idx)
            redox_site_result    = batch1_subgraph3_result[redox_site_idx]
            redox_site_result_   = redox_site_result.unsqueeze(0)

            "gate = sigmoid(FFN(x)) -1, 1"
            redox_site_change = redox_site_result_ * self.gate_GCN3(redox_site_result_) + redox_site_result_

            batch3_redox_idx  = mapping_dict.get(redox_site_idx)

            all_indices = torch.arange(subgraph3_result.shape[0], device=device)
            nonredox_subgraph3_result = subgraph3_result[all_indices != batch3_redox_idx]
            updated_subgraph3_result  = nonredox_subgraph3_result.clone()
            subgraph3_result_ = torch.cat([updated_subgraph3_result[:batch3_redox_idx], redox_site_change, updated_subgraph3_result[batch3_redox_idx:]], dim=0)

            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph3_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

            redox_pos = [redox[redox_site_idx][0]]
            sites = [i for i in range(len(redox)) if redox[i][0] in redox_pos]
            for site in sites:
                if site in redox_sites:
                    redox_sites.remove(site)

        if E12s == []:
            subgraph3_pooled = self.pool(subgraph3_result, batch3)
            if   reaction == 'reduction':
                E12s = self.E12_reg_red(subgraph3_pooled)
            elif reaction == 'oxidation':
                E12s = self.E12_reg_ox(subgraph3_pooled)
            loss_reg = nn.MSELoss()(E12s.squeeze(), real_E12s[0])
            loss_reg.backward(retain_graph=True)

        return loss

    def sample(self, batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, real_E12s, reaction, redox, redox_order = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox, graph.oreder_site

        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2 = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        #"results after GCN and result_ after global pooling"
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

        # convert batch1 index to batch3 index
        m_batch1  = batch1[midx]
        new_batch = batch2[batch1_2.long()]

        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device='cuda')

        batch1_subgraph3_result = subgraph3_result[ordered_indices]

        if   reaction == 'reduction':
            num_redox_all = self.num_peaks_red(batch1_subgraph3_result)
        elif reaction == 'oxidation':
            num_redox_all = self.num_peaks_ox(batch1_subgraph3_result)
        
        num_redox_ = torch.argmax(num_redox_all, dim=1)
        pred_num_redox_ = num_redox_.clone()
        pred_E12s  = torch.tensor([], device=device)
        while num_redox_.sum() != 0:
            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            if   reaction == 'reduction':
                E12s          = self.E12_reg_red(batch1_subgraph3_result)
            elif reaction == 'oxidation':
                E12s          = self.E12_reg_ox(batch1_subgraph3_result)
            E12s       = E12s.squeeze()
            redox_mask = num_redox_ > 0
            # redox_indices = torch.nonzero(redox_mask).squeeze()
            redox_indices = torch.nonzero(redox_mask, as_tuple=False).flatten()
            E12s_redox = E12s[redox_mask]
            if reaction == "reduction":
                E12, filtered_idx = torch.max(E12s_redox, dim=0)
            elif reaction == "oxidation":
                E12, filtered_idx = torch.min(E12s_redox, dim=0)

            redox_idx = redox_indices[filtered_idx].item()
            redox_site_result  = batch1_subgraph3_result[redox_idx]
            redox_site_result_ = redox_site_result.unsqueeze(0)
            redox_site_change  = redox_site_result_ * self.gate_GCN3(redox_site_result_) + redox_site_result_
            
            batch3_redox_idx  = mapping_dict.get(redox_idx)
            all_indices = torch.arange(subgraph3_result.shape[0], device=device)
            nonredox_subgraph3_result = subgraph3_result[all_indices != batch3_redox_idx]
            updated_subgraph3_result  = nonredox_subgraph3_result.clone()
            subgraph3_result_ = torch.cat([updated_subgraph3_result[:batch3_redox_idx], redox_site_change, updated_subgraph3_result[batch3_redox_idx:]], dim=0)

            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph3_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

            pred_E12s = torch.cat((pred_E12s, E12.unsqueeze(0)), 0)

            num_redox_[redox_idx] = num_redox_[redox_idx] - 1
            
        return num_redox_all, pred_num_redox_, pred_E12s

class GCN1_RNN_OMGNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, dropout=0.3, mode='auto'):
        super(OMGNN, self).__init__()
        self.GCN1 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=0.3)
        self.GCN2 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=0.3)
        self.GCN3 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=0.3)
        self.pool = global_mean_pool
        self.potential_reg = nn.Sequential(
            nn.Linear(hidden_dim + 1, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.num_peaks_red = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5),
            nn.Softmax())
        self.num_peaks_ox = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5),
            nn.Softmax())
        self.E12_reg_red = nn.Sequential(
            nn.Linear(hidden_dim , 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.E12_reg_ox = nn.Sequential(
            nn.Linear(hidden_dim , 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.gate_GCN3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())

    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index


    def forward_subgraph(self, x, edge_index, batch, edge_attr, gcn, pre_proc=None, transform_edge_attr=None):
        if pre_proc is not None:
            x = pre_proc(x)

        rev_edge_index = self._rev_edge_index(edge_index)

        if transform_edge_attr is not None:
            edge_attr = transform_edge_attr(edge_attr)

        data = Data(x=x, edge_index=edge_index, rev_edge_index=rev_edge_index, edge_attr=edge_attr)

        if isinstance(gcn, GATConv):
            result = gcn(x, edge_index, edge_attr) 
        else:
            result = gcn(data) 

        result_pooled = self.pool(result, batch)
        return result, result_pooled

    def forward(self, batch):
        """只對redox site的分類loss backward,
        但最後ligands沒有redox site後所有一起backward"""
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, real_E12s, reaction, redox, redox_order = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox, graph.oreder_site

        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2 = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        #"results after GCN and result_ after global pooling"
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))


        total_loss = 0
        # convert batch1 index to batch3 index
        m_batch1  = batch1[midx]
        new_batch = batch2[batch1_2.long()]

        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device='cuda')

        redox_order = redox_order.split('/')
        index_order = []
        for item in redox_order:
            matching_keys = [key for key, values in batch.redox.items() if any(value[0] == item for value in values)]
            if len(matching_keys) == 1:
                index_order.append(matching_keys[0])
            else:
                index_order.append(matching_keys)      

        num_each_redox = [redox[i][1] for i in range(len(redox))]
        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        for i, site_index in enumerate(index_order):
            if type(site_index) is list:
                break
            if  reaction == 'reduction':
                num_redox_site = self.num_peaks_red(batch1_subgraph3_result[site_index])
                E12            = self.E12_reg_red(batch1_subgraph3_result[site_index])
            elif reaction == 'oxidation':
                num_redox_site = self.num_peaks_ox(batch1_subgraph3_result[site_index])
                E12            = self.E12_reg_ox(batch1_subgraph3_result[site_index])
            loss_cla = nn.CrossEntropyLoss()(num_redox_site, torch.tensor(num_each_redox[site_index], device='cuda'))
            loss_reg = nn.MSELoss()(E12.squeeze(), real_E12s[i])
            loss = loss_cla + loss_reg
            loss.backward(retain_graph=True)
            total_loss += loss
            num_each_redox[site_index] = num_each_redox[site_index] - 1

            all_indices = torch.arange(subgraph3_result.shape[0], device='cuda')
            redox_node  = subgraph3_result[site_index].clone()
            redox_node  = redox_node * self.gate_GCN3(redox_node) + redox_node
            batch3_redox_idx  = mapping_dict.get(site_index)
            nonredox_subgraph3_result = subgraph3_result[all_indices != batch3_redox_idx]
            updated_subgraph3_result  = nonredox_subgraph3_result.clone()

            subgraph3_result_ = torch.cat([updated_subgraph3_result[:batch3_redox_idx], redox_node.unsqueeze(0), updated_subgraph3_result[batch3_redox_idx:]], dim=0)
            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph3_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        if   reaction == 'reduction':
            num_redox_all = self.num_peaks_red(batch1_subgraph3_result)
        elif reaction == 'oxidation':
            num_redox_all = self.num_peaks_ox(batch1_subgraph3_result)

        loss_cla = nn.CrossEntropyLoss()(num_redox_all, torch.tensor(num_each_redox, device='cuda'))
        loss = loss_cla
        loss.backward()
        total_loss += loss

        return total_loss 
    

# subgraph1_result, subgraph1_pooled = model.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=model.GCN1)
# subgraph2_result, subgraph2_pooled = model.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=model.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
# subgraph3_result, subgraph3_pooled = model.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=model.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))