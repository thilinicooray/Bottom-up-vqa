import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy

from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), torch.mean(self.attn, 1)


class CAQModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, updated_query_composer, neighbour_attention, Dropout_C, classifier, dataset):
        super(CAQModel, self).__init__()
        self.dataset = dataset
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.updated_query_composer = updated_query_composer
        self.neighbour_attention = neighbour_attention
        self.Dropout_C = Dropout_C
        self.classifier = classifier

    def forward(self, v, b, q, labels, mask):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        #CAQ

        cur_group = joint_repr.contiguous().view(mask.size(0), -1, joint_repr.size(-1))

        neighbours, _ = self.neighbour_attention(cur_group, cur_group, cur_group, mask=mask)

        withctx = neighbours.contiguous().view(v.size(0), -1)

        updated_q_emb = self.Dropout_C(self.updated_query_composer(torch.cat([withctx, q_emb], -1)))

        att = self.v_att(v, updated_q_emb)
        v_emb = (att * v).sum(1)
        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(updated_q_emb)
        joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)
        return logits


def build_caq(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates + 1, 0.5)
    return CAQModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_caq_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    updated_query_composer = FCNet([num_hid * 2, num_hid])
    neighbour_attention = MultiHeadedAttention(4, num_hid//2, dropout=0.1)
    Dropout_C = nn.Dropout(0.1)


    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates + 1, 0.5)
    return CAQModel(w_emb, q_emb, v_att, q_net, v_net, updated_query_composer, neighbour_attention, Dropout_C, classifier, dataset)
