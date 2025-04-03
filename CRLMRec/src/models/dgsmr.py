

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
        
        self.attention_fc = nn.Linear(embedding_dim, 1)

    def forward(self, i_emb_u, v_emb_u, t_emb_u):

        
        att_i = self.attention_fc(i_emb_u).squeeze(-1)
        att_v = self.attention_fc(v_emb_u).squeeze(-1)
        att_t = self.attention_fc(t_emb_u).squeeze(-1)

        
        attention_weights = F.softmax(torch.stack([att_i, att_v, att_t]), dim=0)

        
        att_weights_i = attention_weights[0]
        att_weights_v = attention_weights[1]
        att_weights_t = attention_weights[2]

        
        weighted_i_emb_u = att_weights_i.unsqueeze(-1) * i_emb_u
        weighted_v_emb_u = att_weights_v.unsqueeze(-1) * v_emb_u
        weighted_t_emb_u = att_weights_t.unsqueeze(-1) * t_emb_u

        
        fused_user_embedding =weighted_i_emb_u   +  weighted_t_emb_u + weighted_v_emb_u 
        
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
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers'] 
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight'] 
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight'] 
        self.dropout = config['dropout']#[0.8, 0.9]
        self.degree_ratio = config['degree_ratio']

        self.n_nodes = self.n_users + self.n_items 

        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.masked_adj, self.mm_adj = None, None
        self.image_adj,self.text_adj=None,None

        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)
        

        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.modalityFusion=MultiModalAttention(self.embedding_dim)
        
        self.head_num=4
        self.model_cat_rate=config['model_cat_rate']
        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_k': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_v': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([self.head_num*self.embedding_dim, self.embedding_dim]))),
        })
        self.embedding_dict = {'user':{}, 'item':{}}
        
        
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
        self.ssl_temp = self.config["ssl_temp"]
        self.infonce_criterion = nn.CrossEntropyLoss()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path,'image_adj_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(dataset_path,'text_adj_{}.pt'.format(self.knn_k))
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))
        mul_modal_cnt = 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False) #
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
            mul_modal_cnt += 1
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)
            mul_modal_cnt += 1
        self.item_feat_dim = self.embedding_dim * (mul_modal_cnt + 1)
        
        # 初始化经过GCN后的物品、用户嵌入层
        self.embedding_item_after_GCN = nn.Linear(self.item_feat_dim, self.embedding_dim) 
        self.embedding_user_after_GCN = nn.Linear(self.item_feat_dim, self.embedding_dim) 
        #mean pooling融合
        # self.embedding_item_after_GCN = nn.Linear(self.embedding_dim, self.embedding_dim) 
        # self.embedding_user_after_GCN = nn.Linear(self.embedding_dim, self.embedding_dim) 
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)

        
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        # if os.path.exists(image_adj_file):
        #     self.image_adj = torch.load(image_adj_file)
        # if os.path.exists(text_adj_file):
        #     self.text_adj = torch.load(text_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj=self.image_adj= image_adj
                
                
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj =self.text_adj= text_adj
                
            if self.v_feat is not None and self.t_feat is not None:
                
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
            # torch.save(self.image_adj, image_adj_file)
            # torch.save(self.text_adj, text_adj_file)

    def compute_funsion_adj(self):
        
        alpha = F.softmax(self.attention_weights, dim=0)
        self.mm_adj = alpha[0] * self.image_adj + alpha[1] * self.text_adj
        
        return self.mm_adj,alpha[0]
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    "user-item图去噪操作"

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def contrastive_loss(self, z1, z2,tau=0.5):
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(torch.mm(z1, z1.t()))
        between_sim = f(torch.mm(z1, z2.t()))
        
        loss = -torch.log(
            torch.diag(between_sim) /
            (refl_sim.sum(dim=1) + between_sim.sum(dim=1) - torch.diag(refl_sim))
        ).mean()
        return loss

    def info_nce_loss(self,h_v, h_t, temperature=0.5):
        h_v_norm = F.normalize(h_v, p=2, dim=1)  # [n_items, embed_dim]
        h_t_norm = F.normalize(h_t, p=2, dim=1)  # [n_items, embed_dim]
        
        
        sim_matrix = torch.matmul(h_v_norm, h_t_norm.T) / temperature  # [n_items, n_items]
        
        
        labels = torch.arange(sim_matrix.size(0)).long().to(h_v.device)
        
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def mm_fusion(self, reps: list):
        #concatenation方法
        z = torch.cat(reps, dim=1)
        
        # z = torch.mean(torch.stack(reps), dim=0)
        return z
    def para_dict_to_tenser(self, para_dict):  
        
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors
    
    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):  
        
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
        
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
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
        
        if self.v_feat is not None:
            self.image_feats = self.image_trs(self.image_embedding.weight)#[962,4096]--->[962,64]
        if self.t_feat is not None:
            self.text_feats = self.text_trs(self.text_embedding.weight)#[962,64]
        
        users_emb = self.user_embedding.weight 
        items_emb = self.item_id_embedding.weight 

        
        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            embs = [all_emb]
            g_droped = adj
            for _ in range(self.n_ui_layers):#self.n_ui_layers
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            return light_out 

        self.i_emb = compute_graph(users_emb, items_emb)
        self.i_emb_u, self.i_emb_i = torch.split(self.i_emb, [self.n_users, self.n_items])#patio[1686, 64]  [962, 64]
        
        self.v_emb = compute_graph(users_emb, self.image_feats)
        self.v_emb_u, self.v_emb_i = torch.split(self.v_emb, [self.n_users, self.n_items])#patio[1686, 64]  [962, 64]
        
        
        self.t_emb = compute_graph(users_emb, self.text_feats)
        self.t_emb_u, self.t_emb_i = torch.split(self.t_emb, [self.n_users, self.n_items])#patio[1686, 64]  [962, 64]
    

        all_users=self.modalityFusion(self.i_emb_u, self.v_emb_u, self.t_emb_u)
        all_items=self.modalityFusion(self.i_emb_i, self.v_emb_i, self.t_emb_i)
        
        # self.t_sne_for_modality( self.v_emb_i, self.t_emb_i)
        
        return all_users, all_items + h

        
    def t_sne_for_modality_yuan(self,id_emb, v_emb, t_emb):
        
        i_emb = id_emb.cpu().detach().numpy() 
        v_emb = v_emb.cpu().detach().numpy()
        t_emb = t_emb.cpu().detach().numpy()
        
        all_embs = concatenate([i_emb, v_emb, t_emb], axis=0)
        
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        
        tsne_results = tsne.fit_transform(all_embs)
        
        center_i = np.mean(tsne_results[:len(i_emb)], axis=0)
        center_v = np.mean(tsne_results[len(i_emb):len(i_emb)+len(v_emb)], axis=0)
        center_t = np.mean(tsne_results[len(i_emb)+len(v_emb):], axis=0)
        
        plt.figure(figsize=(16,10))
        # plt.scatter(tsne_results[:len(i_emb),0], tsne_results[:len(i_emb),1], label='ID Embeddings')
        plt.scatter(tsne_results[len(i_emb):len(i_emb)+len(v_emb),0], tsne_results[len(i_emb):len(i_emb)+len(v_emb),1], label='Visual Embeddings', marker='^')
        plt.scatter(tsne_results[len(i_emb)+len(v_emb):,0], tsne_results[len(i_emb)+len(v_emb):,1], label='Text Embeddings', c='r')

        
        plt.scatter(center_v[0], center_v[1], s=200, c='black', marker='*')
        plt.scatter(center_t[0], center_t[1], s=200, c='black', marker='*')
        plt.legend()
        output_dir = '/home/bjf/bjf_projects/MMRec/src/tsne/no_cl/patio'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.savefig(os.path.join(output_dir, f'patio_cl-t-sne_plot_{timestamp}.svg'), format="svg")
        # plt.show()
        
        plt.close()

    
    def t_sne_for_modality(self, v_emb, t_emb):
        
        selected_indices = [10, 100, 150, 200, 300]
        v_emb = v_emb.cpu().detach().numpy()
        t_emb = t_emb.cpu().detach().numpy()
        
        v_emb_selected = v_emb[selected_indices]
        t_emb_selected = t_emb[selected_indices]
        
        combined_emb = np.vstack((v_emb_selected, t_emb_selected))
        
        tsne = TSNE(n_components=2, random_state=0, perplexity=5)
        emb_2d = tsne.fit_transform(combined_emb)
        
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
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.savefig(os.path.join(output_dir, f'patio-t-sne_plot_{timestamp}.svg'), format="svg")
        # plt.show()
        
        plt.close()



    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)#, image_item_embeds, text_item_embeds, fusion_embs
        self.build_item_graph = False
        
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        "BPR Loss"
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)  
        mf_v_loss, mf_t_loss , contrastive_loss= 0.0, 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
        
        
        # contrastive_loss+=self.contrastive_loss(text_item_embeds,fusion_embs)
        # contrastive_loss+=self.contrastive_loss(image_item_embeds,fusion_embs)
        "新增对比损失"
        contrastive_loss=self.info_nce_loss(text_feats,image_feats)
        
        '自监督损失，users,pos_items'
        pos=pos_items
        ssl_loss = self.compute_ssl(users, pos)
        
        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)+ self.config['ssl_alpha'] * ssl_loss# + self.config['ssl_alpha']*contrastive_loss 
    
    def compute_ssl(self, users, items):
        return self.fac(items)
    
    def fac(self, idx):
        
        x_i_iv = self.g_i_iv(self.i_emb_i[idx])
        x_v_iv = self.g_v_iv(self.v_emb_i[idx])
        v_logits = torch.mm(x_i_iv, x_v_iv.T)
        v_logits /= self.ssl_temp
        v_labels = torch.tensor(list(range(x_i_iv.shape[0]))).to(self.device)
        v_loss = self.infonce_criterion(v_logits, v_labels)

        
        x_iv_iva = self.g_iv_iva(x_i_iv)
        x_iva_ivat = self.g_iva_ivat(x_iv_iva)
        x_t_ivat = self.g_t_ivat(self.t_emb_i[idx])

        t_logits = torch.mm(x_iva_ivat, x_t_ivat.T)
        t_logits /= self.ssl_temp
        t_labels = torch.tensor(list(range(x_iva_ivat.shape[0]))).to(self.device)
        t_loss = self.infonce_criterion(t_logits, t_labels)

        #return v_loss + a_loss + t_loss
        return v_loss + t_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    
    def create_adj_mat(self, interaction_csr):
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
