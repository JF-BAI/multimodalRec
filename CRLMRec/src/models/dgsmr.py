
#修改之后的模型
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian

from numpy import concatenate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

class MultiModalAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(MultiModalAttention, self).__init__()
        self.embedding_dim = embedding_dim
        # 定义一个全连接层来计算注意力权重
        self.attention_fc = nn.Linear(embedding_dim, 1)

    def forward(self, i_emb_u, v_emb_u, t_emb_u):

        # 计算每个模态的注意力权重
        att_i = self.attention_fc(i_emb_u).squeeze(-1)
        att_v = self.attention_fc(v_emb_u).squeeze(-1)
        att_t = self.attention_fc(t_emb_u).squeeze(-1)

        # 使用softmax函数归一化注意力权重
        attention_weights = F.softmax(torch.stack([att_i, att_v, att_t]), dim=0)

        # 将注意力权重切分成原来的三部分
        att_weights_i = attention_weights[0]
        att_weights_v = attention_weights[1]
        att_weights_t = attention_weights[2]

        # 进行加权融合
        weighted_i_emb_u = att_weights_i.unsqueeze(-1) * i_emb_u
        weighted_v_emb_u = att_weights_v.unsqueeze(-1) * v_emb_u
        weighted_t_emb_u = att_weights_t.unsqueeze(-1) * t_emb_u

        # 融合所有模态的信息
        fused_user_embedding =weighted_i_emb_u   +  weighted_t_emb_u + weighted_v_emb_u 
        # 保存权重用于后续绘图
        self.weights = attention_weights.detach().cpu().numpy()

        return fused_user_embedding
class DGSMR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DGSMR, self).__init__(config, dataset)
        self.config = config
        self.embedding_dim = config['embedding_size'] #64
        self.feat_embed_dim = config['feat_embed_dim']#64
        self.knn_k = config['knn_k']#10
        self.lambda_coeff = config['lambda_coeff']#0.9
        self.cf_model = config['cf_model']#未使用
        self.n_layers = config['n_mm_layers']#1 item-item图层数
        self.n_ui_layers = config['n_ui_layers']#2user-item图层数
        self.reg_weight = config['reg_weight'] #列表[0.0, 1e-05, 1e-04, 1e-03]
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight'] #视觉模态item-item图权重0.1，文本模态item-item图权重为1-mm_image_weight
        self.dropout = config['dropout']#[0.8, 0.9]
        self.degree_ratio = config['degree_ratio']#未使用

        self.n_nodes = self.n_users + self.n_items 

        # load dataset info# 加载并预处理数据集信息，user-item图的边信息
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)# 将数据集中的交互矩阵转换为COO格式的稀疏矩阵，并确保其数据类型为32位浮点数
        self.norm_adj = self.get_norm_adj_mat().to(self.device)# 将标准化后的邻接矩阵转移到指定的设备上 #26495*26495 user数+item数
        '目前的问题：'

        # '新增创建二分图邻接矩阵，用于构建模态特定的二分图'
        # train_interactions = dataset.inter_matrix(form='csr').astype(np.float32)
        # coo = self.create_adj_mat(train_interactions).tocoo()# 调用self中的方法create_adj_mat来创建邻接矩阵，并将其转换为COO格式
        # indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])# 将COO格式的邻接矩阵的行和列索引转换为PyTorch的LongTensor格式
        # self.original_norm_adj = torch.sparse.FloatTensor(indices, torch.FloatTensor(coo.data), coo.shape)# 使用PyTorch的sparse.FloatTensor方法，将处理过的邻接矩阵转换为稀疏张量格式
        # self.original_norm_adj = self.original_norm_adj.to(self.device)# 将稀疏邻接矩阵张量移动到指定的设备（如GPU）#26495*26495 user数+item数
        



        self.masked_adj, self.mm_adj = None, None# 初始化掩码邻接矩阵和模态混合模型邻接矩阵
        self.image_adj,self.text_adj=None,None

        self.edge_indices, self.edge_values = self.get_edge_info()# 获取图的边信息，包括边的索引和值
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)# 将边的索引和值移动到指定的设备（如GPU）上
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)# 生成一个完整的边索引数组，用于后续操作
        # self.masked_adj=self.compute_pagerank_centrality()#pagerank中心性剪枝方法
        # self.masked_adj=self.pre_epoch_processing()
        

        "初始化user/item id的嵌入层 ，64维"
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        '注意力机制融合多模态'
        "初始化注意力机制所需的权重参数"
        self.modalityFusion=MultiModalAttention(self.embedding_dim)
        

        self.head_num=4
        self.model_cat_rate=config['model_cat_rate']
        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({# 初始化注意力机制中的权重参数字典
            'w_q': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_k': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_v': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([self.head_num*self.embedding_dim, self.embedding_dim]))),
        })
        self.embedding_dict = {'user':{}, 'item':{}}# 初始化用户和项目的嵌入字典 # 这里尚未对嵌入进行具体初始化，只创建了空字典结构
        
        '模态特定的自监督任务相关初始化'
        self.g_i_iv = nn.Linear(self.embedding_dim, self.embedding_dim)#id->id.v 64->64
        self.g_v_iv = nn.Linear(self.embedding_dim, self.embedding_dim)#v->id.v 64->64
        self.g_iv_iva = nn.Linear(self.embedding_dim, self.embedding_dim)#iv->id.v.a 不需要 64->64
        self.g_a_iva = nn.Linear(self.embedding_dim, self.embedding_dim)#a->id.v.a 不需要 64->64
        self.g_iva_ivat = nn.Linear(self.embedding_dim, self.embedding_dim // 2)#id.v.a->id.v.a.t  64->32
        self.g_t_ivat = nn.Linear(self.embedding_dim, self.embedding_dim // 2)#t->id.v.a.t  64->32
        nn.init.xavier_uniform_(self.g_i_iv.weight)
        nn.init.xavier_uniform_(self.g_v_iv.weight)
        nn.init.xavier_uniform_(self.g_iv_iva.weight)
        nn.init.xavier_uniform_(self.g_a_iva.weight)
        nn.init.xavier_uniform_(self.g_iva_ivat.weight)
        nn.init.xavier_uniform_(self.g_t_ivat.weight)
        self.ssl_temp = self.config["ssl_temp"]#自监督任务的温度参数
        self.infonce_criterion = nn.CrossEntropyLoss()#交叉熵损失

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])#获取数据集路径
        image_adj_file = os.path.join(dataset_path,'image_adj_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(dataset_path,'text_adj_{}.pt'.format(self.knn_k))
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))# 构造多模态邻接矩阵文件名：根据knn_k值和图像权重生成特定的文件名，存储路径为dataset_path（数据集文件夹）下
        "使用原始的模态特征构建item-item图,然后将两个模态下的item图按权重进行混合"
        mul_modal_cnt = 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False) #
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
            mul_modal_cnt += 1# 增加多模态计数器，表示在模型中已经处理了另一种模态的数据
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)
            mul_modal_cnt += 1# 增加多模态计数器，表示在模型中已经处理了另一种模态的数据
        self.item_feat_dim = self.embedding_dim * (mul_modal_cnt + 1)
        
        # 初始化经过GCN后的物品、用户嵌入层
        self.embedding_item_after_GCN = nn.Linear(self.item_feat_dim, self.embedding_dim) #转换维度 64 * 模态数--->64
        self.embedding_user_after_GCN = nn.Linear(self.item_feat_dim, self.embedding_dim) #转换维度 64 * 模态数--->64
        #mean pooling融合
        # self.embedding_item_after_GCN = nn.Linear(self.embedding_dim, self.embedding_dim) #转换维度 64 * 模态数--->64
        # self.embedding_user_after_GCN = nn.Linear(self.embedding_dim, self.embedding_dim) #转换维度 64 * 模态数--->64
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)

        # 尝试加载多模态邻接矩阵，如果不存在则计算并保存
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        # if os.path.exists(image_adj_file):
        #     self.image_adj = torch.load(image_adj_file)
        # if os.path.exists(text_adj_file):
        #     self.text_adj = torch.load(text_adj_file)
        else:#检查哪一种模态特征存在，若存在则用来创建多模态邻接矩阵
            if self.v_feat is not None:# 如果视觉特征存在，计算图像的K近邻邻接矩阵
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj=self.image_adj= image_adj
                
                
            if self.t_feat is not None:# 如果文本特征存在，计算文本的K近邻邻接矩阵
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj =self.text_adj= text_adj
                
            if self.v_feat is not None and self.t_feat is not None:# 如果视觉和文本特征都存在，按权重组合两个邻接矩阵
                # '使用超参数来融合两个模态的邻接矩阵'
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)#将多模态邻接矩阵保存到指定的文件中
            # torch.save(self.image_adj, image_adj_file)
            # torch.save(self.text_adj, text_adj_file)

    def compute_funsion_adj(self):
        # 计算多模态邻接矩阵的融合权重
        alpha = F.softmax(self.attention_weights, dim=0)
        self.mm_adj = alpha[0] * self.image_adj + alpha[1] * self.text_adj
        
        return self.mm_adj,alpha[0]
    def get_knn_adj_mat(self, mm_embeddings):#输入经过预训练得到的模态嵌入向量，计算模态的K近邻邻接矩阵
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))# 标准化嵌入向量，使其长度为1，以便计算余弦相似度
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))# 计算所有项目嵌入之间的余弦相似度
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)# 获取每个项目嵌入的K个最近邻的索引
        adj_size = sim.size()
        del sim# 释放相似度矩阵，以节省内存
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)# 生成一个与knn_ind相同行数的索引数组，用于后续操作
        indices0 = torch.unsqueeze(indices0, 1)# 将索引数组的形状从(n,)变为(n, 1)，以方便后续扩展
        indices0 = indices0.expand(-1, self.knn_k)# 将索引数组在第二维度上扩展至knn_k大小，准备与knn_ind进行广播操作
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)# 将扩展后的indices0和knn_ind按指定方式组合，形成稀疏矩阵的索引
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)# 返回索引和计算得到的标准化拉普拉斯矩阵

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):#获取标准化的邻接矩阵，首先构建user与item的二分图邻接矩阵，然后标准化，最后转换为系数张量
        A = sp.dok_matrix((self.n_users + self.n_items,# 创建了一个尺寸为 (用户数 + 物品数, 用户数 + 物品数) 的稀疏矩阵 A
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix# 用户-物品交互矩阵及其转置
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),# 构建邻接矩阵的数据字典
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)# 更新邻接矩阵
        # norm adj matrix# 计算标准化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor将标准化邻接矩阵转换为PyTorch稀疏张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    "user-item图去噪操作"
    "基于度敏感的边裁剪"
    def pre_epoch_processing(self):#未使用  在每个训练轮次前进行处理
        if self.dropout <= .0:# 当dropout比率小于等于0时，将规范化的邻接矩阵赋值给masked_adj
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning # 度敏感的边裁剪
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))# 根据dropout比率计算需要保留的边的数量 ，保留 |总边数|*（1-dropout）
        #通过torch.multinomial函数根据每条边的权重（edge_values）按概率进行采样。这意味着权重较大的边更有可能被保留下来，这体现了度敏感的特点。
        degree_idx = torch.multinomial(self.edge_values, degree_len)# 从edge_values中按概率随机采样degree_len个索引，这些索引代表保留的边，获取保留边的索引，被选中的边用来构建新的邻接矩阵，
        # random sample# # 从原始的边索引中选择保留的边索引
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values# 对保留的边进行权重规范化
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))# 将keep_values连接起来，因为keep_values是用户和物品的权重，这里将其合并
        # update keep_indices to users/items+self.n_users# 更新keep_indices，将物品ID偏移用户的数量，以区分用户和物品
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)
        return self.masked_adj#原本是没有的


    "基于随机游走(PageRank)的中心性的边裁剪PageRank centrality"
    def compute_pagerank_centrality(self, tau=0.5, alpha=0.85, max_iter=100, tol=1e-6):#tau=0.5,
        # 1. 构建用户-物品二分图的邻接矩阵 (COO格式)
        num_edges = self.edge_indices.shape[1]
        adj_matrix = torch.sparse.FloatTensor(self.edge_indices.to(self.device), self.edge_values.to(self.device), torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])).to_dense().to(self.device)

        # 2. 计算节点的PageRank中心性
        N = self.n_users + self.n_items
        degree = adj_matrix.sum(dim=1)  # 计算度
        pagerank = torch.ones(N,device=self.device) / N  # 初始化PageRank值，设为均匀分布
        teleport = pagerank.clone().to(self.device)  # teleport向量，均匀随机跳转

        # 随机游走迭代计算PageRank
        for _ in range(max_iter):
            prev_pagerank = pagerank.clone()
            pagerank = (1 - alpha) * teleport + alpha * adj_matrix.matmul(pagerank / degree)  # PageRank更新公式
            if torch.norm(pagerank - prev_pagerank, p=1) < tol:
                break

        # 3. 计算边的中心性 (用户和物品节点分别计算)
        user_pagerank = pagerank[:self.n_users]  # 用户节点的PageRank值
        item_pagerank = pagerank[self.n_users:]  # 物品节点的PageRank值

        edge_centralities = []
        for i in range(num_edges):
            user_idx = self.edge_indices[0, i]
            item_idx = self.edge_indices[1, i] - self.n_users  # 物品索引需要减去用户的数量偏移
            # 确保物品索引是有效的
            if item_idx >= 0 and item_idx < self.n_items:
                edge_centrality = (user_pagerank[user_idx] + item_pagerank[item_idx]) / 2  # 边的中心性
                edge_centralities.append(edge_centrality)
            else:
                print(f"Warning: Invalid item index {item_idx} for edge {i}")

        edge_centralities = torch.tensor(edge_centralities)

        # 4. 判断噪声边并保留高于阈值的边
        keep_mask = edge_centralities >= tau  # 保留中心性大于阈值的边
        keep_indices = torch.stack([self.edge_indices[0][keep_mask], self.edge_indices[1][keep_mask]])
        keep_values = self.edge_values[keep_mask]

        # 5. 用保留的边构建新的邻接矩阵
        self.masked_adj = torch.sparse_coo_tensor(keep_indices, keep_values, torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])).to(self.device)

        return self.masked_adj

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def contrastive_loss(self, z1, z2,tau=0.5):#该函数的目的是通过对比增强两个输入表示z1和z2之间的差异性。
        f = lambda x: torch.exp(x / tau)#tau= 0.5
        refl_sim = f(torch.mm(z1, z1.t()))
        between_sim = f(torch.mm(z1, z2.t()))
        # 计算对比损失
        loss = -torch.log(
            torch.diag(between_sim) /
            (refl_sim.sum(dim=1) + between_sim.sum(dim=1) - torch.diag(refl_sim))
        ).mean()
        return loss

    def info_nce_loss(self,h_v, h_t, temperature=0.5):
        """
        计算基于互信息的对比学习损失（InfoNCE Loss）。
        
        参数:
        - h_v: 视觉模态的表示 [n_items, embed_dim]
        - h_t: 文本模态的表示 [n_items, embed_dim]
        - temperature: 温度参数，默认为 0.5
        
        返回:
        - loss: InfoNCE 损失值
        """
        
        # 正则化嵌入向量为单位向量
        h_v_norm = F.normalize(h_v, p=2, dim=1)  # [n_items, embed_dim]
        h_t_norm = F.normalize(h_t, p=2, dim=1)  # [n_items, embed_dim]
        
        # 计算所有物品的视觉和文本嵌入之间的余弦相似度
        sim_matrix = torch.matmul(h_v_norm, h_t_norm.T) / temperature  # [n_items, n_items]
        
        # 标签：正样本在对角线上（同一物品的两种模态）
        labels = torch.arange(sim_matrix.size(0)).long().to(h_v.device)
        
        # InfoNCE 损失（交叉熵损失，其中正样本是对角线上的样本）
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def _normalize_adj_m(self, indices, adj_size):#返回表转化后的邻接矩阵的值
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):#通过user-item交互矩阵获取边信息
        rows = torch.from_numpy(self.interaction_matrix.row)#提取行和列的索引，并转为张量
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)# 构建边的二维张量，包含行和列索引
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))# 根据边的索引和用户与物品数量，对边进行归一化处理
        return edges, values

    def mm_fusion(self, reps: list):#多模态融合 输入：包含不同模态表示的列表  返回 融合后的多模态表示
        #concatenation方法
        z = torch.cat(reps, dim=1)#在1维上拼接 64*模态数
        #平均池化方法
        # z = torch.mean(torch.stack(reps), dim=0)
        return z
    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        """
        将参数字典中的张量合并为一个单一的张量堆栈。

        参数:
            para_dict (nn.ParameterDict): 一个包含多个张量的字典。

        返回:
            tensor: 一个将所有输入张量沿第一个维度堆叠的单一张量。
        """
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors
    '定义注意力机制函数，'
    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):  
        """
        实现多头自注意力机制。

        参数:
        trans_w (dict): 包含权重矩阵的字典，用于变换查询（Q）、键（K）和值（V）。
        embedding_t_1 (Tensor): 时间步t-1的嵌入向量。
        embedding_t (Tensor): 时间步t的嵌入向量。

        返回:
        Z (Tensor): 经过多头自注意力机制处理后的输出向量。
        att (Tensor): 自注意力机制的注意力权重矩阵。
        """#
        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], self.embedding_dim/self.head_num

        Q = torch.matmul(q, trans_w['w_q']) 
        K = torch.matmul(k, trans_w['w_k'])
        V = v

        Q = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)
        K = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2) 
        K = torch.unsqueeze(K, 1)  
        V = torch.unsqueeze(V, 1)  

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  
        att = torch.sum(att, dim=-1) 
        att = torch.unsqueeze(att, dim=-1)  
        att = F.softmax(att, dim=2)  

        Z = torch.mul(att, V)  
        Z = torch.sum(Z, dim=2)  

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])

        self.model_cat_rate*F.normalize(Z, p=2, dim=2)
        return Z, att.detach()
    
    def forward(self, adj):
        'item-item图卷积获得物品嵌入'
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):#多模态item-item图卷积，获得物品的嵌入向量表示
            h = torch.sparse.mm(self.mm_adj, h)
        # image_item_embeds=self.item_id_embedding.weight
        # text_item_embeds =self.item_id_embedding.weight
        # for i in range(self.n_layers):
        #     image_item_embeds=torch.sparse.mm(self.image_adj, image_item_embeds)
        # for i in range(self.n_layers):
        #     text_item_embeds=torch.sparse.mm(self.text_adj, text_item_embeds )
        # # "融合模态"
        # att = torch.cat([self.query(image_item_embeds), self.query(text_item_embeds)], dim=-1)
        # weight = self.softmax(att)
        # h = weight[:, 0].unsqueeze(dim=1) * image_item_embeds + weight[:, 1].unsqueeze(dim=1) * text_item_embeds
        #adj 形状[2648, 2648]，是稀疏张量
        "转换特征向量的维度"
        if self.v_feat is not None:
            self.image_feats = self.image_trs(self.image_embedding.weight)#[962,4096]--->[962,64]
        if self.t_feat is not None:
            self.text_feats = self.text_trs(self.text_embedding.weight)#[962,64]
        '获得用户和物品ID的嵌入'
        users_emb = self.user_embedding.weight #用户id嵌入 64维
        items_emb = self.item_id_embedding.weight #物品id嵌入 64维

        '定义二分图卷积操作函数，获得用户和物品的表示'
        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            embs = [all_emb]
            g_droped = adj#将去噪后的user-item图
            for _ in range(self.n_ui_layers):#self.n_ui_layers
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            return light_out #返回用户和物品的表示
        '创建模态特定的二分图，并通过GNN获得用户和项目的表示，三种模态（id模态，视觉模态，文本模态）'#self.n_users + self.n_items
        #id模态
        self.i_emb = compute_graph(users_emb, items_emb)
        self.i_emb_u, self.i_emb_i = torch.split(self.i_emb, [self.n_users, self.n_items])#patio[1686, 64]  [962, 64]
        
        #visual模态
        self.v_emb = compute_graph(users_emb, self.image_feats)
        self.v_emb_u, self.v_emb_i = torch.split(self.v_emb, [self.n_users, self.n_items])#patio[1686, 64]  [962, 64]
        
        #text模态
        self.t_emb = compute_graph(users_emb, self.text_feats)
        self.t_emb_u, self.t_emb_i = torch.split(self.t_emb, [self.n_users, self.n_items])#patio[1686, 64]  [962, 64]
        
        
        # # 多模态融合方法一：concat拼接
        # all_users = self.embedding_user_after_GCN(self.mm_fusion([self.i_emb_u, self.v_emb_u, self.t_emb_u]))#维度 64 * 模态数--->64   #1429, 64
        # all_items = self.embedding_item_after_GCN(self.mm_fusion([self.i_emb_i, self.v_emb_i, self.t_emb_i]))#维度 64 * 模态数--->64   #900, 64
        

        # # 多模态融合方法二：attention
        all_users=self.modalityFusion(self.i_emb_u, self.v_emb_u, self.t_emb_u)
        all_items=self.modalityFusion(self.i_emb_i, self.v_emb_i, self.t_emb_i)
        

        # modality_weights=[]
        # numbers=[23,198,245]
        # for i in numbers:
        #     #第一个
        #     modality_weights.append(self.modalityFusion.weights[0][i])
        #     modality_weights.append(self.modalityFusion.weights[1][i])
        #     modality_weights.append(self.modalityFusion.weights[2][i])

        # weights_array = np.array(modality_weights).reshape(3, 3)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(weights_array, annot=True, fmt='.3f', cmap='YlGnBu',
        #     xticklabels=['ID modality', 'Visual modality', 'Text modality'],
        #     yticklabels=['23', '198', '245'])
        # plt.title('Heat map of user weights for different modalities')
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_path = f'/home/bjf/bjf_projects/MMRec/src/heatemp/music/heatmap_attention_weight_music_{timestamp}.png'  # 指定要保存的文件路径
        # plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        # # user_ids = [100,200]
        # # self.plot_heatmap(user_ids)

        # self.i_emb_i.shape  [900,64]
        # self.v_emb_i.shape  [900,64]
        # self.t_emb_i.shape  [900,64]
        '融合前的项目的多模态向量self.i_emb_i, self.v_emb_i, self.t_emb_i,均位于gpu上'
        # self.t_sne_for_modality_yuan( self.i_emb_i,self.v_emb_i, self.t_emb_i)#T-SNE画图
        #个例研究
        self.t_sne_for_modality( self.v_emb_i, self.t_emb_i)#T-SNE画图
        
        return all_users, all_items + h

        
        """
        #user_embedding.weight [1686, 64]      self.item_id_embedding.weight [962, 64]
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)# 将用户嵌入矩阵和物品嵌入矩阵拼接成一个大的嵌入矩阵，在行上进行拼接
        all_embeddings = [ego_embeddings]# 初始化只包含原始嵌入的列表
        for i in range(self.n_ui_layers):# 通过user-item交互图进行卷积，更新嵌入向量
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)# 将所有层的嵌入向量平均，得到最终的嵌入向量
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        # 将嵌入向量拆分为用户嵌入和物品嵌入
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h,# image_item_embeds, text_item_embeds, h  #得到用户的最终嵌入和物品的嵌入向量
        """
    '使用T-SNE进行可视化'


    #原
    def t_sne_for_modality_yuan(self,id_emb, v_emb, t_emb):
        # 假设这些向量已经在GPU上
        i_emb = id_emb.cpu().detach().numpy() # 转换为NumPy数组并在CPU上处理
        v_emb = v_emb.cpu().detach().numpy()
        t_emb = t_emb.cpu().detach().numpy()
        # 拼接所有模态的嵌入向量
        all_embs = concatenate([i_emb, v_emb, t_emb], axis=0)
        # 初始化t-SNE模型
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        # 对嵌入向量进行降维
        tsne_results = tsne.fit_transform(all_embs)
        # 计算每个模态的聚类中心
        center_i = np.mean(tsne_results[:len(i_emb)], axis=0)
        center_v = np.mean(tsne_results[len(i_emb):len(i_emb)+len(v_emb)], axis=0)
        center_t = np.mean(tsne_results[len(i_emb)+len(v_emb):], axis=0)
        #绘制模态散点
        plt.figure(figsize=(16,10))
        # plt.scatter(tsne_results[:len(i_emb),0], tsne_results[:len(i_emb),1], label='ID Embeddings')
        plt.scatter(tsne_results[len(i_emb):len(i_emb)+len(v_emb),0], tsne_results[len(i_emb):len(i_emb)+len(v_emb),1], label='Visual Embeddings', marker='^')
        plt.scatter(tsne_results[len(i_emb)+len(v_emb):,0], tsne_results[len(i_emb)+len(v_emb):,1], label='Text Embeddings', c='r')

        #绘制聚类中心
        plt.scatter(center_v[0], center_v[1], s=200, c='black', marker='*')
        plt.scatter(center_t[0], center_t[1], s=200, c='black', marker='*')
        plt.legend()
        output_dir = '/home/bjf/bjf_projects/MMRec/src/tsne/no_cl/patio'
        # 创建目录如果不存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 获取当前日期和时间，用于生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存图片到指定目录
        plt.savefig(os.path.join(output_dir, f'patio_cl-t-sne_plot_{timestamp}.svg'), format="svg")
        # plt.show()
        # 关闭图形以释放内存
        plt.close()

    #最新case study
    def t_sne_for_modality(self, v_emb, t_emb):
        # 选择特定的5个项目
        selected_indices = [10, 100, 150, 200, 300]

        # 假设这些向量已经在GPU上
        # 转换为NumPy数组并在CPU上处理
        v_emb = v_emb.cpu().detach().numpy()
        t_emb = t_emb.cpu().detach().numpy()
        # 获取这5个项目的视觉和文本模态的嵌入表示
        v_emb_selected = v_emb[selected_indices]
        t_emb_selected = t_emb[selected_indices]
        # 将两个模态的数据合并在一起准备进行t-SNE降维
        combined_emb = np.vstack((v_emb_selected, t_emb_selected))
        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2, random_state=0, perplexity=5)
        emb_2d = tsne.fit_transform(combined_emb)
        # 分离出两个模态的数据点
        v_emb_2d = emb_2d[:len(selected_indices)]
        t_emb_2d = emb_2d[len(selected_indices):]
        # 可视化
        plt.figure(figsize=(8, 6))
        plt.scatter(v_emb_2d[:, 0], v_emb_2d[:, 1], c='r', label='Visual Modality')
        plt.scatter(t_emb_2d[:, 0], t_emb_2d[:, 1], c='b', label='Textual Modality')

        for i in range(len(selected_indices)):#len(selected_indices)
            plt.text(v_emb_2d[i, 0], v_emb_2d[i, 1], str(selected_indices[i]), fontsize=7, color='black', alpha=0.7)
            plt.text(t_emb_2d[i, 0], t_emb_2d[i, 1], str(selected_indices[i]), fontsize=7, color='black', alpha=0.7)
        
        plt.legend()

        output_dir = '/home/bjf/bjf_projects/MMRec/src/case_study/no_cl/patio'
        # 创建目录如果不存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 获取当前日期和时间，用于生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存图片到指定目录
        plt.savefig(os.path.join(output_dir, f'patio-t-sne_plot_{timestamp}.svg'), format="svg")
        # plt.show()
        # 关闭图形以释放内存
        plt.close()



    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        # 从交互数据中解包用户、正样本物品和负样本物品
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        # 通过图神经网络模型计算用户和物品的嵌入表示
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)#, image_item_embeds, text_item_embeds, fusion_embs
        self.build_item_graph = False## 重置build_item_graph标志，以避免重复构建物品图，这是不是就是冻结item-item图？
        # 根据用户和物品的ID，提取相应的嵌入表示
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        "BPR Loss"
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)  #公式（10）前半部分
        mf_v_loss, mf_t_loss , contrastive_loss= 0.0, 0.0, 0.0
        if self.t_feat is not None:# 如果存在文本特征，则计算文本模态下的BPR损失
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:# 如果存在视觉特征，则计算视觉模态下的BPR损失
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
        
        
        # contrastive_loss+=self.contrastive_loss(text_item_embeds,fusion_embs)
        # contrastive_loss+=self.contrastive_loss(image_item_embeds,fusion_embs)
        "新增对比损失"
        contrastive_loss=self.info_nce_loss(text_feats,image_feats)#互信息之跨模态对比损失，鼓励相同物品在不同模态下的表示接近，不同物品的模态表示拉远
        
        '自监督损失，users,pos_items'
        pos=pos_items
        ssl_loss = self.compute_ssl(users, pos)# 计算SSL损失，通过自我监督学习策略进一步优化模型泛化能力。
        #能否在此处进行
        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)+ self.config['ssl_alpha'] * ssl_loss# + self.config['ssl_alpha']*contrastive_loss #计算BPR损失，user与正负样本之间的bpr损失+  超参数lamda +不同模态下的BPR损失
    
    def compute_ssl(self, users, items):
        return self.fac(items)#调用模态特定的FAC自监督互信息损失函数计算（多层次粒度空间）
    
    def fac(self, idx):#FAC：模态特定自监督任务；计算多模态特征对齐的互信息损失函数 输入：items=interaction[1],正样本ID
        # 视觉模态的自监督互信息损失计算
        x_i_iv = self.g_i_iv(self.i_emb_i[idx])#self.i_emb_i：id模态二分图卷积所得物品表示，将正样本物品的嵌入的维度转换为隐藏维度
        x_v_iv = self.g_v_iv(self.v_emb_i[idx])#self.v_emb_i：v模态二分图卷积所得物品表示，将正样本物品的嵌入的维度转换为隐藏维度
        v_logits = torch.mm(x_i_iv, x_v_iv.T)# 计算相似度
        v_logits /= self.ssl_temp## 温度缩放
        v_labels = torch.tensor(list(range(x_i_iv.shape[0]))).to(self.device)# 生成标签
        v_loss = self.infonce_criterion(v_logits, v_labels)# 计算视觉模态的InfoNCE损失

        
        x_iv_iva = self.g_iv_iva(x_i_iv)
        x_iva_ivat = self.g_iva_ivat(x_iv_iva)
        x_t_ivat = self.g_t_ivat(self.t_emb_i[idx])

        t_logits = torch.mm(x_iva_ivat, x_t_ivat.T)
        t_logits /= self.ssl_temp
        t_labels = torch.tensor(list(range(x_iva_ivat.shape[0]))).to(self.device)
        t_loss = self.infonce_criterion(t_logits, t_labels)#计算文本模态的InfoNCE损失

        #return v_loss + a_loss + t_loss
        return v_loss + t_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    
    def create_adj_mat(self, interaction_csr):#创建二分图的邻接矩阵self.n_users + self.n_items
        user_np, item_np = interaction_csr.nonzero()
        # user_list, item_list = self.dataset.get_train_interactions()
        # user_np = np.array(user_list, dtype=np.int32)
        # item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.n_users + self.n_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_items)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        adj_type = self.config['adj_type']
        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1)) + 1e-08    # avoid RuntimeWarning: divide by zero encountered in power
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix
