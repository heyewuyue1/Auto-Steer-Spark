import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


@variational_estimator
class Prediction(nn.Module):

    def __init__(self,
                 in_feature=69,
                 hid_units=256,
                 contract=1,
                 mid_layers=True,
                 res_con=True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con

        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units // contract)
        # self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)
        self.mid_mlp2 = BayesianLinear(hid_units // contract, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):

        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out


class FeatureEmbed(nn.Module):
    def __init__(self, embed_size=32, tables = 35, types=18, joins = 133, columns= 429, \
                 ops=8, pos=4, bin_number = 50):
        super(FeatureEmbed, self).__init__()

        self.embed_size = embed_size
        self.bin_number = bin_number

        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)

        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size // 8)
        self.posEmbed = nn.Embedding(pos, embed_size // 8)

        self.linearFilter2 = nn.Linear(embed_size + embed_size // 8 + 1,
                                       embed_size + embed_size // 8 + 1)
        self.linearFilter = nn.Linear(embed_size + embed_size // 8 + 1,
                                      embed_size + embed_size // 8 + 1)

        self.linearType = nn.Linear(embed_size, embed_size)

        self.linearJoin = nn.Linear(embed_size, embed_size)

        # self.linearSample = nn.Linear(1000, embed_size)

        self.linearHist = nn.Linear(bin_number, embed_size)

        self.joinEmbed = nn.Embedding(joins, embed_size)

        self.project = nn.Linear(
            embed_size * 4 + embed_size // 8 + 1 + embed_size // 8,
            embed_size * 4 + embed_size // 8 + 1 + embed_size // 8)

    # input: B by 14 (type, join, f1, f2, f3, mask1, mask2, mask3)
    def forward(self, feature):

        # typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(feature,(1,1,9,3,self.bin_number*3,1001), dim = -1)
        typeId, joinId, filtersId, filtersMask, table_sample, posId = torch.split(
            feature, (1, 1, 60, 20, 1, 1), dim=-1)

        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)

        # histEmb = self.getHist(hists, filtersMask)
        tableEmb = self.getTable(table_sample)
        posEmb = self.getPos(posId)

        final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, posEmb),
                          dim=1)
        final = F.leaky_relu(self.project(final))

        return final

    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())

        return emb.squeeze(1)

    def getTable(self, table_sample):
        table = table_sample
        emb = self.tableEmbed(table.long()).squeeze(1)

        return emb

    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())

        return emb.squeeze(1)

    def getPos(self, posId):
        emb = self.posEmbed(posId.long())

        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        # batch * 50 * 3
        histExpand = hists.view(-1, self.bin_number, 3).transpose(1, 2)

        emb = self.linearHist(histExpand)
        emb[~filtersMask.bool()] = 0.  # mask out space holder

        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(emb, dim=1)
        avg = total / num_filters.view(-1, 1)

        return avg

    def getFilter(self, filtersId, filtersMask):
        ## get Filters, then apply mask
        torch.set_printoptions(profile="full")
        torch.set_printoptions(sci_mode=False)
        # print(filtersId[0])
        # print(filtersMask[:100])
        # print('filterId shape:',filtersId.shape)
        filterExpand = filtersId.view(-1, 3, 20)
        # print('filterExpand shape:',filterExpand.shape)
        # print(filterExpand[0])
        # filterExpand = filtersId.view(-1, 3, 3).transpose(1, 2)
        colsId = filterExpand[:, 0, :].long()
        opsId = filterExpand[:, 1, :].long()
        vals = filterExpand[:, 2, :].unsqueeze(-1)  # b by 3 by 1

        # b by 3 by embed_dim
        # print('colsId shape:',colsId.shape)
        try:
            col = self.columnEmbed(colsId)
        except:
            print(colsId)

        
        # print('opsId shape:',opsId.shape)
        try:
            op = self.opEmbed(opsId)
        except:
            print(opsId)
        concat = torch.cat((col, op, vals), dim=-1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))

        ## apply mask
        concat[~filtersMask.bool()] = 0.

        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(concat, dim=1)
        # print(num_filters.view(-1, 1))
        avg = total / (num_filters.view(-1, 1) + 1e-8)

        return avg


#     def get_output_size(self):
#         size = self.embed_size * 5 + self.embed_size // 8 + 1
#         return size


class Dose(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 bin_number = 50, \
                 pred_hid = 256
                ):

        super(Dose, self).__init__()

        hidden_dim = emb_size * 4 + emb_size // 8 + emb_size // 8 + emb_size // 8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.emb_size = emb_size
        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, emb_size // 8, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        encoders = [
            EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate,
                         head_size) for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)

        self.embbed_layer = FeatureEmbed(emb_size, bin_number=bin_number)

        self.pred = Prediction(hidden_dim, pred_hid)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)

    def forward(self, batched_data):
        
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x

        heights = batched_data.heights

        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(
            1, self.head_size, 1, 1)

        # rel pos
        # rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        # tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # reset rel pos here
        # t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        # tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        # tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        tree_attn_bias = tree_attn_bias[:, :, 1:, 1:]
        # print(x.shape)
        # x_view = x.view(-1, 16)
        x_view = x.view(-1, 84)

        node_feature = self.embbed_layer(x_view).view(
            n_batch, -1, self.hidden_dim - self.emb_size // 8)
        
        # -1 is number of dummy
        # node_feature = node_feature + self.height_encoder(heights)
        height_feature = self.height_encoder(heights)
        node_feature = torch.cat([node_feature, height_feature], dim=2)
        # super_node_feature = node_feature[:,0,:]
        # super_node_feature_2 = node_feature[:,-1,:]
        # super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        # super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)
        
        # transfomrer encoder
        output = self.input_dropout(node_feature)
        # print(output)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        return self.pred(output[:, 0, :])  #, self.pred2(output[:,0,:])


class FeedForwardNetwork(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size**-0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        # print(attn_bias)
        if attn_bias is not None:
            # x = x + attn_bias
            x = x * attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size,
                                                 attention_dropout_rate,
                                                 head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):

        y = self.self_attention_norm(x)
        # print(y)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
