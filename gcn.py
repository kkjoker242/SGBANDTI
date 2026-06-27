import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import dgl
import dgl.nn as dglnn
from torch_scatter import scatter_mean


class NestedGCN(nn.Module):
    """
    NestedGCN模型，适配DGL格式
    支持subgraph_to_graph的两层池化操作
    """
    def __init__(self, input_dim, num_layers, hidden, num_classes = None, use_z=False, use_rd=False):
        """
        Args:
            input_dim: 输入特征维度
            num_layers: GCN层数
            hidden: 隐藏层维度
            num_classes: 输出类别数
            use_z: 是否使用节点标签嵌入
            use_rd: 是否使用阻力距离
        """
        super(NestedGCN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        
        if self.use_rd:
            self.rd_projection = nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = nn.Embedding(1000, 8)
        
        # 调整输入维度
        adjusted_input_dim = input_dim
        if self.use_z or self.use_rd:
            adjusted_input_dim += 8

        # GCN层
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(dglnn.GraphConv(adjusted_input_dim, hidden, activation=F.relu))
        for i in range(num_layers - 1):
            self.conv_layers.append(dglnn.GraphConv(hidden, hidden, activation=F.relu))
        
        # # 分类层
        self.lin1 = nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        for conv in self.conv_layers:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        # self.lin2.reset_parameters()

    def forward(self, graph, x):
        """
        Args:
            graph: DGL图对象，必须包含：
                - ndata['h']: 节点特征
                - ndata['node_to_subgraph']: 节点到子图的映射
                - subgraph_to_graph: 子图到图的映射（作为图的属性或ndata）
        
        Returns:
            log_softmax输出
        """
        # 获取节点特征
        # x = graph.ndata['h']
        
        # 节点标签嵌入（如果启用）
        z_emb = 0
        if self.use_z and 'z' in graph.ndata:
            z_emb = self.z_embedding(graph.ndata['z'])
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        # 阻力距离嵌入（如果启用）
        if self.use_rd and 'rd' in graph.ndata:
            rd_proj = self.rd_projection(graph.ndata['rd'])
            z_emb += rd_proj
        
        # 拼接特征
        if self.use_rd or self.use_z:
            x = torch.cat([z_emb, x], -1)
        
        # 临时存储特征到图的ndata中，用于DGL的池化操作
        graph.ndata['feat'] = x
        
        # GCN前向传播，收集所有层的输出
        xs = []
        for i, conv in enumerate(self.conv_layers):
            x = conv(graph, x)
            xs.append(x)
        
        # 拼接所有层的输出
        x = torch.cat(xs, dim=1)
        
        # 第一层池化：从节点到子图
        if 'node_to_subgraph' in graph.ndata:
            node_to_subgraph = graph.ndata['node_to_subgraph']
            # 使用scatter_mean进行子图级别的池化
            num_subgraphs = node_to_subgraph.max().item() + 1
            x_subgraph = scatter_mean(x, node_to_subgraph, dim=0, dim_size=num_subgraphs)
        else:
            # 如果没有node_to_subgraph，直接使用图级别池化
            x_subgraph = dgl.mean_nodes(graph, 'feat')
            if x_subgraph.dim() == 1:
                x_subgraph = x_subgraph.unsqueeze(0)


        x_graph = self.lin1(x_subgraph)
        
        return x_graph

    def __repr__(self):
        return self.__class__.__name__


class GCN(nn.Module):
    """
    标准GCN模型，适配DGL格式
    """
    def __init__(self, input_dim, num_layers, hidden, num_classes):
        super(GCN, self).__init__()
        
        # GCN层
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(dglnn.GraphConv(input_dim, hidden, activation=F.relu))
        for i in range(num_layers - 1):
            self.conv_layers.append(dglnn.GraphConv(hidden, hidden, activation=F.relu))
        
        # 分类层
        self.lin1 = nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, graph):
        """
        Args:
            graph: DGL图对象，包含ndata['h']节点特征
        """
        x = graph.ndata['h']
        
        # 临时存储特征到图的ndata中
        graph.ndata['feat'] = x
        
        # GCN前向传播
        xs = []
        for conv in self.conv_layers:
            x = conv(graph, x)
            xs.append(x)
        
        # 拼接所有层的输出
        x = torch.cat(xs, dim=1)
        
        # 更新临时特征
        graph.ndata['feat'] = x
        
        # 图级别池化
        x = dgl.mean_nodes(graph, 'feat')
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        
        
        return x

    def __repr__(self):
        return self.__class__.__name__
