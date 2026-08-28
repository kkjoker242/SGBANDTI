import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_min
from gcn import NestedGCN, MolecularGCNStd
from ban import BANLayer as BCNet
from torch.nn.utils.weight_norm import weight_norm


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    # 模型输出层已含 sigmoid，这里不再施加，避免双重 sigmoid
    n = torch.squeeze(pred_output, 1)
    loss = loss_fct(n, labels)
    return n, loss


class SGBANDTI(nn.Module):
    def __init__(self, **config):
        super(SGBANDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        mlp_dropout = config["DECODER"]["DROPOUT"]
        drug_max_nodes = config["DRUG"]["MAX_NODES"]
        drug_num_layers = config["DRUG"]["NUM_LAYERS"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.use_subgraph = config["ABLATION"]["USE_SUBGRAPH"]
        self.use_ban = config["ABLATION"]["USE_BAN"]

        if self.use_subgraph:
            self.drug_extractor = MolecularGCN(
                in_feats=drug_in_feats,
                dim_embedding=drug_embedding,
                hidden_feats=drug_hidden_feats,
                max_nodes=drug_max_nodes,
                num_layers=drug_num_layers
            )
        else:
            # 消融：标准 GCN（整分子消息传递 + 真实节点 mean 池化）
            self.drug_extractor = MolecularGCNStd(
                in_feats=drug_in_feats,
                dim_embedding=drug_embedding,
                hidden=drug_hidden_feats,
                num_layers=drug_num_layers
            )
        self.protein_extractor = ProteinCNN(
            protein_emb_dim,
            num_filters,
            kernel_size,
            protein_padding
        )

        if self.use_ban:
            self.bcn = weight_norm(
                BCNet(
                    v_dim=drug_hidden_feats,
                    q_dim=num_filters[-1],
                    h_dim=mlp_in_dim,
                    h_out=ban_heads
                ),
                name='h_mat',
                dim=None
            )
        else:
            # 消融：concat+MLP 融合（替代双线性注意力）
            self.fusion = ConcatFusion(
                v_dim=drug_hidden_feats,
                q_dim=num_filters[-1],
                out_dim=mlp_in_dim
            )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary, dropout=mlp_dropout)

    def forward(self, bg_d, v_p, mode="train"):
        v_d, v_mask = self.drug_extractor(bg_d)
        # 蛋白真实长度 = 非零编码数；卷积后输出位置 p 覆盖输入 [p, p+15]，
        # 仅当 p >= 真实长度时为纯 padding，应屏蔽（q 维掩码）
        protein_len = (v_p != 0).sum(dim=1)
        v_p = self.protein_extractor(v_p)
        q_num = v_p.size(1)
        q_mask = (torch.arange(q_num, device=v_p.device)[None, :] < protein_len[:, None]).float()
        if self.use_ban:
            f, att = self.bcn(v_d, v_p, v_mask=v_mask, q_mask=q_mask)
        else:
            f = self.fusion(v_d, v_mask, v_p, q_mask)
            att = None
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, hidden_feats=None, max_nodes=290, num_layers=3):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.max_nodes = max_nodes
        self.gnn = NestedGCN(input_dim=dim_embedding, hidden=hidden_feats, num_layers=num_layers)
        self.output_feats = hidden_feats

    def forward(self, batch_graph):
        # 用索引读取而不是 pop，避免修改输入图（同一图被多次 forward 时不会丢 'h'）
        node_feats = batch_graph.ndata['h']
        # 指示位列（最后一维）：虚拟节点=1、真实节点=0，据此得到逐节点真实掩码
        real_flag = (node_feats[:, -1] < 0.5).float()
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = int((batch_graph.batch_size) / self.max_nodes)
        feats = node_feats.view(batch_size, -1, self.output_feats)
        # 逐子图真实掩码：虚拟根节点的子图仅含自身（real=0），真实根节点子图全为真实节点（real=1）。
        # 虚拟节点不连真实节点，故用 scatter_min 取子图内最小值即可区分。
        node_to_subgraph = batch_graph.ndata['node_to_subgraph']
        num_subgraphs = node_to_subgraph.max().item() + 1
        subgraph_real, _ = scatter_min(real_flag, node_to_subgraph, dim=0, dim_size=num_subgraphs)
        mask = subgraph_real.view(batch_size, -1)
        return feats, mask


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class ConcatFusion(nn.Module):
    """消融用（USE_BAN=False）：药物/蛋白 masked mean 池化后拼接，再线性投影。"""
    def __init__(self, v_dim, q_dim, out_dim):
        super(ConcatFusion, self).__init__()
        self.proj = nn.Linear(v_dim + q_dim, out_dim)

    def forward(self, v, v_mask, q, q_mask):
        v_pool = (v * v_mask.unsqueeze(-1)).sum(1) / v_mask.sum(1).unsqueeze(-1).clamp(min=1)
        q_pool = (q * q_mask.unsqueeze(-1)).sum(1) / q_mask.sum(1).unsqueeze(-1).clamp(min=1)
        return self.proj(torch.cat([v_pool, q_pool], dim=1))


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout=0.0):
        super(MLPDecoder, self).__init__()
        # 注意: p=0 时 dropout 仍会消耗 RNG（污染 DataLoader shuffle），
        # 故用 self.use_drop 门控，p=0 时完全跳过以保证默认配置逐位可复现。
        self.use_drop = dropout > 0
        self.drop = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        if self.use_drop:
            x = self.drop(self.bn1(F.relu(self.fc1(x))))
            x = self.drop(self.bn2(F.relu(self.fc2(x))))
            x = self.drop(self.bn3(F.relu(self.fc3(x))))
        else:
            x = self.bn1(F.relu(self.fc1(x)))
            x = self.bn2(F.relu(self.fc2(x)))
            x = self.bn3(F.relu(self.fc3(x)))
        # 输出层 sigmoid，与论文表述一致，输出为交互概率
        x = torch.sigmoid(self.fc4(x))
        return x
