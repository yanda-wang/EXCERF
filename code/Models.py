import dill
import datetime
import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from Parameters import Parameters

parameters = Parameters()


# ====================topological transformer，利用token之间的关系进行计算==============
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ff_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ff_size, hidden_size)
        self.initialize_weight()

    def initialize_weight(self, init_range=0.1):
        self.layer1.weight.data.uniform_(-init_range, init_range)
        self.layer2.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadSelfAttn(nn.Module):
    def __init__(self, hidden_size, attn_dropout_rate, head_num):
        super(MultiHeadSelfAttn, self).__init__()

        self.hidden_size = hidden_size
        self.attn_dropout_rate = attn_dropout_rate
        self.head_num = head_num

        self.attn_size = hidden_size // head_num
        self.scale = self.attn_size ** -0.5

        self.linear_query = nn.Linear(self.hidden_size, self.head_num * self.attn_size)
        self.linear_key = nn.Linear(self.hidden_size, self.head_num * self.attn_size)
        self.linear_value = nn.Linear(self.hidden_size, self.head_num * self.attn_size)
        self.attn_dropout = nn.Dropout(self.attn_dropout_rate)

        self.output_layer = nn.Linear(self.head_num * self.attn_size, self.hidden_size)
        self.initialize_weight()

    def initialize_weight(self, init_range=0.1):
        self.linear_query.weight.data.uniform_(-init_range, init_range)
        self.linear_key.weight.data.uniform_(-init_range, init_range)
        self.linear_value.weight.data.uniform_(-init_range, init_range)
        self.output_layer.weight.data.uniform_(-init_range, init_range)

    def forward(self, query, key, value, attn_bias=None):
        original_query_size = query.size()
        dimension_key = self.attn_size
        dimension_value = self.attn_size
        node_count = query.size(0)

        # 处理数据，并划分到不同的head
        query = self.linear_query(query).view(-1, self.head_num, dimension_key)  # size=(node_count,head_num,attn_size)
        key = self.linear_key(key).view(-1, self.head_num, dimension_key)
        value = self.linear_value(value).view(-1, self.head_num, dimension_value)

        query = query.transpose(0, 1)  # size=(head_num,node_count,attn_size)
        key = key.transpose(0, 1).transpose(1, 2)  # size=(head_num,attn_size,node_count)
        value = value.transpose(0, 1)  # size=(head_num,node_count,attn_size)

        query = query * self.scale
        x = torch.matmul(query, key)  # size=(head_num,node_count,node_count),token之间的交互信息，计算self-attention的基础

        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=2)  # size=(head_num,node_count,node_count)
        x = x.matmul(value)  # size=(head_num,node_count,attn_size)
        x = x.transpose(0, 1).contiguous()  # size=(node_count,head_num,attn_size)
        x = x.view(-1, self.head_num * self.attn_size)  # size=(node_count,(head_num*attn_size))
        x = self.output_layer(x)

        assert x.size() == original_query_size
        return x


class TransEncoderLayer(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_rate, attn_dropout_rate, head_num):
        super(TransEncoderLayer, self).__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadSelfAttn(hidden_size, attn_dropout_rate, head_num)
        self.self_attn_dropout = nn.Dropout(dropout_rate)

        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = FeedForwardNetwork(hidden_size, ff_size)
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias):
        y = self.self_attn_norm(x)  # size=(node_count,hidden_size)
        y = self.self_attn(y, y, y, attn_bias)
        y = self.self_attn_dropout(y)
        x = x + y

        y = self.ff_norm(x)
        y = self.ff(y)
        y = self.ff_dropout(y)
        x = x + y

        return x


# ================global memory====================
class GlobalMemory(nn.Module):
    def __init__(self, hidden_size, memory_num):
        super(GlobalMemory, self).__init__()
        self.hidden_size = hidden_size
        self.memory_num = memory_num

        self.linear_memory_key = nn.Linear(self.hidden_size, self.memory_num, bias=False)
        self.linear_memory_value = nn.Linear(self.memory_num, self.hidden_size, bias=False)
        self.output_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.initialize_weight()

    def initialize_weight(self, init_range=0.1):
        self.linear_memory_key.weight.data.uniform_(-init_range, init_range)
        self.linear_memory_value.weight.data = self.linear_memory_key.weight.data.permute(1, 0)
        self.output_layer.weight.data.uniform_(-init_range, init_range)

    def forward(self, query):
        """
        :param query: size=(batch_size,1,hidden_size),这里的batch_size其实也就是病人对应的admission数量
        :return:
        """
        x = self.linear_memory_key(query)  # size=(batch_size,1,memory_num)
        attn = torch.softmax(x, dim=2)  # size=(batch_size,1,memory_num)
        x = self.linear_memory_value(attn)  # size=(batch_size,1,hidden_size)
        x = self.output_layer(x)
        x = x + query
        x = F.relu(x)

        return x


# ==========================GCN===========================

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, device, item_count, embedding_size, adj_matrix, dropout_rate):
        super(GCN, self).__init__()
        self.device = device
        self.item_count = item_count
        self.embedding_size = embedding_size

        adj_matrix = self.normalize(adj_matrix + np.eye(adj_matrix.shape[0]))
        self.adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        self.x = torch.eye(item_count).to(self.device)

        self.gcn1 = GraphConvolution(item_count, embedding_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gcn2 = GraphConvolution(embedding_size, embedding_size)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj_matrix)  # dim=(item_count,embedding*size)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj_matrix)  # dim=(item_count,embedding_size)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not an appropriate attention method, choose from dot, general, and concat.")

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    # score=query.T()*keys
    def dot_score(self, query, keys):
        return torch.sum(query * keys, -1).unsqueeze(0)  # dim=(1,keys.dim(0))

    # score=query.T()*W*keys, W is a matrix
    def general_score(self, query, keys):
        energy = self.attn(keys)
        return torch.sum(query * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0))

    # score=v.T()*tanh(W*[query;keys])
    def concat_score(self, query, keys):
        energy = self.attn(torch.cat((query.expand(keys.size(0), -1), keys), -1)).tanh()
        return torch.sum(self.v * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0)

    def initialize_weights(self, init_range):
        if self.method == 'concat':
            self.v.data.uniform_(-init_range, init_range)

    def forward(self, query, keys):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(query, keys)
        elif self.method == 'concat':
            attn_energies = self.concat_score(query, keys)
        else:  # dot
            attn_energies = self.dot_score(query, keys)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1)  # dim=(1,keys.dim(0))


# =========================药物推荐模型========================

class Encoder(nn.Module):
    def __init__(self, hidden_size, code_encoding_dropout_rate, trans_layer_num_sub, trans_dropout_rate,
                 trans_attn_dropout_rate, trans_head_num, global_memory_num, gru_n_layers, gru_dropout_rate,
                 graph_file=parameters.GRAPH_FILE, voc_file=parameters.VOC_FILE, device=parameters.DEVICE):
        super(Encoder, self).__init__()
        self.device = device
        dill_file = open(voc_file, 'rb')
        self.voc = dill.load(dill_file)
        dill_file.close()
        self.diagnoses_count = self.voc.get_diagnoses_count()
        self.procedures_count = self.voc.get_procedures_count()
        self.medications_count = self.voc.get_medications_count()

        self.diagnoses_embedding = nn.Embedding(self.diagnoses_count, hidden_size)
        self.procedures_embeddings = nn.Embedding(self.procedures_count, hidden_size)
        self.diagnoses_embedding.weight.data.uniform_(-0.1, 0.1)
        self.procedures_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.code_encoding_dropout = nn.Dropout(p=code_encoding_dropout_rate)

        dill_file = open(graph_file, 'rb')
        graph = dill.load(dill_file)
        dill_file.close()
        self.distance_graph_diagnoses = graph['distance_graph_diagnoses']
        self.weight_graph_diagnoses = graph['weight_graph_diagnoses']
        self.distance_graph_procedures = graph['distance_graph_procedures']
        self.weight_graph_procedures = graph['weight_graph_procedures']

        self.distance_embedding_diagnoses = nn.Embedding(1, trans_head_num)
        self.distance_embedding_procedures = nn.Embedding(1, trans_head_num)

        ff_size = hidden_size * 4
        trans_encoder_diagnoses = [
            TransEncoderLayer(hidden_size, ff_size, trans_dropout_rate, trans_attn_dropout_rate, trans_head_num)
            for _ in range(trans_layer_num_sub)]
        self.trans_encoder_diagnoses = nn.ModuleList(trans_encoder_diagnoses)
        trans_encoder_procedures = [
            TransEncoderLayer(hidden_size, ff_size, trans_dropout_rate, trans_attn_dropout_rate, trans_head_num)
            for _ in range(trans_layer_num_sub)]
        self.trans_encoder_procedures = nn.ModuleList(trans_encoder_procedures)

        self.global_memory_diagnoses = GlobalMemory(hidden_size, global_memory_num)
        self.global_memory_procedures = GlobalMemory(hidden_size, global_memory_num)

        self.gru = nn.GRU(hidden_size, hidden_size, gru_n_layers,
                          dropout=(0 if gru_n_layers == 1 else gru_dropout_rate), bidirectional=False)

    def get_graph(self, idx, distance_graph, weight_graph):
        distance_graph = distance_graph[idx, :]
        distance_graph = distance_graph[:, idx]
        weight_graph = weight_graph[idx, :]
        weight_graph = weight_graph[:, idx]
        distance_graph = torch.Tensor(distance_graph).to(self.device)
        weight_graph = torch.Tensor(weight_graph).to(self.device)

        return distance_graph, weight_graph

    def forward(self, patient):
        seq_diagnoses = []
        seq_procedures = []
        memory_values = []

        for adm in patient:
            diagnoses, procedures, medications = adm[parameters.DIAGNOSE_INDEX], adm[parameters.PROCEDURE_INDEX], adm[
                parameters.MEDICATION_INDEX]
            _, weight_graph_diagnoses = self.get_graph(diagnoses, self.distance_graph_diagnoses,
                                                       self.weight_graph_diagnoses)
            _, weight_graph_procedures = self.get_graph(procedures, self.distance_graph_procedures,
                                                        self.weight_graph_procedures)
            # +++++++++++处理疾病信息，基于疾病之间的拓扑结构计算transformer+++++++++++
            edge_info_diagnoses = torch.zeros_like(weight_graph_diagnoses, dtype=torch.long)  # size=(#diag,#diag)
            # size=(#diagnoses,hidden_size)
            trans_output_diagnoses = self.diagnoses_embedding(torch.LongTensor(diagnoses).to(self.device))
            trans_output_diagnoses = self.code_encoding_dropout(trans_output_diagnoses)
            for encoder_layer in self.trans_encoder_diagnoses:
                # size=(#diag,#diag,head_num) -> (head_num,#diag,#diag)
                edge_embedding_diagnoses = self.distance_embedding_diagnoses(edge_info_diagnoses).permute(2, 0, 1)
                # size=(head_num,#diag,#diag)
                attn_bias_diagnoses = torch.mul(weight_graph_diagnoses, edge_embedding_diagnoses)
                trans_output_diagnoses = encoder_layer(trans_output_diagnoses, attn_bias_diagnoses)
            seq_diagnoses.append(trans_output_diagnoses.mean(dim=0, keepdim=True))

            # +++++++++++处理procedure信息，基于procedure之间的拓扑结构计算transformer+++++++++++
            edge_info_procedures = torch.zeros_like(weight_graph_procedures, dtype=torch.long)  # size=(#pro,#pro)
            # size=(#procedures,hidden_size)
            trans_output_procedures = self.procedures_embeddings(torch.LongTensor(procedures).to(self.device))
            trans_output_procedures = self.code_encoding_dropout(trans_output_procedures)
            for encoder_layer in self.trans_encoder_procedures:
                # size=(#pro,#pro,head_num) -> (head_num,#pro,#pro)
                edge_embedding_procedures = self.distance_embedding_procedures(edge_info_procedures).permute(2, 0, 1)
                # size=(head_num,#pro,#pro)
                attn_bias_procedures = torch.mul(weight_graph_procedures, edge_embedding_procedures)
                trans_output_procedures = encoder_layer(trans_output_procedures, attn_bias_procedures)
            seq_procedures.append(trans_output_procedures.mean(dim=0, keepdim=True))

            memory_values.append(medications)

        seq_diagnoses = torch.cat(seq_diagnoses).unsqueeze(dim=1)  # size=(#adm,1,hidden_size)
        seq_procedures = torch.cat(seq_procedures).unsqueeze(dim=1)  # size=(#adm,1,hidden_size)
        seq_diagnoses = self.global_memory_diagnoses(seq_diagnoses)
        seq_procedures = self.global_memory_procedures(seq_procedures)
        seq_codes = (seq_diagnoses + seq_procedures) / 2  # size=(#adm,1,hidden_size)
        output_codes, hidden_codes = self.gru(seq_codes)
        queries = output_codes.squeeze(dim=1)  # size=(#adm,hidden_size)
        query = queries[-1:]  # size=(1,hidden_size)

        if len(patient) > 1:
            memory_keys = queries[:-1]  # size=(#adm-1,hidden_size)
            memory_values = memory_values[:-1]  # list，长度为#adm-1，每个元素是对应adm的药品列表
        else:
            memory_keys = None
            memory_values = None

        return query, memory_keys, memory_values


class Decoder(nn.Module):
    def __init__(self, hidden_size, attn_type_kv, attn_type_embedding, medications_ehr_dropout_rate, multi_hop_count,
                 graph_file=parameters.GRAPH_FILE, voc_file=parameters.VOC_FILE, device=parameters.DEVICE):
        super(Decoder, self).__init__()
        self.device = device
        dill_file = open(voc_file, 'rb')
        self.voc = dill.load(dill_file)
        dill_file.close()
        self.diagnoses_count = self.voc.get_diagnoses_count()
        self.procedures_count = self.voc.get_procedures_count()
        self.medications_count = self.voc.get_medications_count()
        self.output_size = self.medications_count
        self.multi_hop_count = multi_hop_count

        dill_file = open(graph_file, 'rb')
        graph = dill.load(dill_file)
        dill_file.close()
        self.ehr_graph = graph['co-occurrence_graph']
        self.ehr_gcn = GCN(self.device, self.medications_count, hidden_size, self.ehr_graph,
                           medications_ehr_dropout_rate)
        self.attn_kv = Attn(attn_type_kv, hidden_size)
        self.attn_embedding = Attn(attn_type_embedding, hidden_size)
        self.output_layer = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size * 3, hidden_size * 2), nn.ReLU(),
                                          nn.Linear(hidden_size * 2, self.output_size))

    def forward(self, query, memory_keys, memory_values):
        if memory_keys is None:
            embedding_medications = self.ehr_gcn()
            weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(weights_embedding, embedding_medications)
            context_o = context_e
        else:
            memory_values_multi_hot = np.zeros((len(memory_values), self.output_size))
            for idx, admission in enumerate(memory_values):
                memory_values_multi_hot[idx, admission] = 1
            memory_values_multi_hot = torch.FloatTensor(memory_values_multi_hot).to(self.device)
            embedding_medications = self.ehr_gcn()
            attn_weights_kv = self.attn_kv(query, memory_keys)
            attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
            read_context = torch.mm(attn_values_kv, embedding_medications)
            update_query = torch.add(query, read_context)
            last_query = update_query

            for hop in range(1, self.multi_hop_count):
                embedding_medications = self.ehr_gcn()
                attn_weights_kv = self.attn_kv(last_query, memory_keys)
                attn_values_kv = attn_weights_kv.mm(memory_values_multi_hot)
                read_context = torch.mm(attn_values_kv, embedding_medications)
                update_query = torch.add(last_query, read_context)
                last_query = update_query

            embedding_medications = self.ehr_gcn()
            attn_weights_embedding = self.attn_embedding(query, embedding_medications)
            context_e = torch.mm(attn_weights_embedding, embedding_medications)
            context_o = last_query

        output = self.output_layer(torch.cat((query, context_o, context_e), -1))  # size=(1,output_size)
        return output
