import torch
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.optim as optim


def nx_to_pyg_data(g: nx.DiGraph) -> Data:
    """
    将带有节点嵌入的 NetworkX 图转换为 PyTorch Geometric 的 Data 对象。

    Args:
        g: 带有 'embedding' 属性的 NetworkX DiGraph。

    Returns:
        PyTorch Geometric 的 Data 对象。
    """
    # 提取节点特征 (x)
    # 确保所有节点都有 'embedding' 属性，并且它们是 Tensor 类型
    node_embeddings = []
    node_id_to_idx = {node_id: i for i, node_id in enumerate(g.nodes())}

    for node_id in g.nodes():
        if 'embedding' not in g.nodes[node_id]:
            # 如果某个节点没有 embedding，可以考虑用零向量填充
            # 或者抛出错误，具体取决于你的业务逻辑
            print(f"Warning: Node {node_id} has no 'embedding' attribute. Using zero vector.")
            node_embeddings.append(torch.zeros(g.nodes[next(iter(g.nodes()))]['embedding_dim']))
        else:
            embedding = g.nodes[node_id]['embedding']
            if isinstance(embedding, torch.Tensor):
                node_embeddings.append(embedding)
            else:
                # 确保 embedding 是 Tensor 类型
                node_embeddings.append(torch.tensor(embedding, dtype=torch.float32))

    x = torch.stack(node_embeddings)

    # 提取边索引 (edge_index)
    edge_indices = []
    for u, v, data in g.edges(data=True):
        src_idx = node_id_to_idx[u]
        dst_idx = node_id_to_idx[v]
        edge_indices.append([src_idx, dst_idx])

    # PyG 的 edge_index 是 [2, num_edges] 的形状
    edge_index = torch.tensor(edge_indices, dtype=torch.long).T

    # 提取边特征 (edge_attr)，如果有的话
    # 在你的图中，边只有 'relation' 属性，这是一个字符串，
    # 如果要作为特征，需要进行独热编码或其他数值化处理。
    # 这里我们暂时不处理边特征，如果需要，可以扩展。
    edge_attr = None # 或者根据需要构建

    # 也可以添加其他图级别的属性，例如：
    # data.num_nodes = G.number_of_nodes()
    # data.num_edges = G.number_of_edges()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 你可能还需要保存原始的 node_id 到 PyG 内部索引的映射，以便后续结果解析
    data.node_id_to_idx = node_id_to_idx
    data.idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}

    return data




class GNNModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # GCNConv 是 PyTorch Geometric 中 GCN 层
        # in_channels: 输入节点特征的维度 (即 SHARED_EMBEDDING_DIM)
        # hidden_channels: 隐藏层的维度
        # out_channels: 输出层维度 (取决于你的任务，例如节点分类的类别数)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 第一层 GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x) # 通常在 GNN 层之后使用激活函数
        x = F.dropout(x, p=0.5, training=self.training) # dropout 用于防止过拟合

        # 第二层 GCN
        x = self.conv2(x, edge_index)

        return x


def train_gnn(g, pyg_data: Data, num_classes: int):
    # 1. 定义模型
    # 输入维度是 SHARED_EMBEDDING_DIM
    # 隐藏层维度可以自定义，例如 128
    # 输出维度是分类任务的类别数
    model = GNNModel(in_channels=512, hidden_channels=128, out_channels=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss() # 节点分类常用交叉熵损失

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pyg_data.to(device)

    # 准备标签 (假设你的节点数据中有 'label' 属性)
    # 你需要根据你的具体任务和数据来构建 'y'
    # 例如：
    # 如果所有节点都有标签，可以遍历 pyg_data.idx_to_node_id 获取原始G图的节点ID，然后从G.nodes[node_id]['label']获取标签
    # 假设你已经有了标签 tensor y
    # pyg_data.y = torch.tensor([G.nodes[pyg_data.idx_to_node_id[i]]['label'] for i in range(len(pyg_data.idx_to_node_id))], dtype=torch.long).to(device)

    # ！！！ 重要：这里只是一个示例，你需要根据你的任务实际构建 pyg_data.y 和 训练/验证/测试掩码
    # 比如，你可以为 'text_chunk' 节点赋予标签，为 'page' 节点赋予标签，等等
    # 假设我们只想对 'text_chunk' 节点进行分类
    labels = []
    train_mask = [] # 用于训练的节点索引
    val_mask = []   # 用于验证的节点索引
    test_mask = []  # 用于测试的节点索引

    # 示例标签和掩码构建 (你需要根据你的实际任务来定义标签和数据集划分)
    # 假设我们只是简单地随机分配标签和掩码进行演示
    # 真实的场景中，你会从G.nodes[node_id]['some_task_label']中获取
    for i, node_id in pyg_data.idx_to_node_id.items():
        node_type = g.nodes[node_id]['type']
        if node_type == 'text_chunk':
            labels.append(0) # 示例：所有文本块节点都属于类别 0
            train_mask.append(True) # 假设所有文本块都用于训练
        elif node_type == 'image':
            labels.append(1) # 示例：所有图片节点都属于类别 1
            train_mask.append(False) # 假设图片节点不参与分类
        else: # page 节点
            labels.append(2) # 示例：所有页面节点都属于类别 2
            train_mask.append(False) # 假设页面节点不参与分类

        # 为了演示，这里简化处理，实际需要更复杂的划分
        val_mask.append(False)
        test_mask.append(False)

    pyg_data.y = torch.tensor(labels, dtype=torch.long).to(device)
    pyg_data.train_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)
    pyg_data.val_mask = torch.tensor(val_mask, dtype=torch.bool).to(device)
    pyg_data.test_mask = torch.tensor(test_mask, dtype=torch.bool).to(device)

    # 训练循环
    print("\n--- 开始 GNN 训练 ---")
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        optimizer.zero_grad() # 梯度清零

        # 前向传播
        out = model(pyg_data.x, pyg_data.edge_index)

        # 计算损失 (只在训练集上计算)
        # out[pyg_data.train_mask] 筛选出训练节点的输出
        # pyg_data.y[pyg_data.train_mask] 筛选出训练节点的真实标签
        loss = criterion(out[pyg_data.train_mask], pyg_data.y[pyg_data.train_mask])

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 评估 (可选，在验证集上)
        model.eval() # 设置模型为评估模式
        with torch.no_grad():
            val_out = model(pyg_data.x, pyg_data.edge_index)
            # 例如计算准确率
            predicted_labels = val_out.argmax(dim=1)
            correct_predictions = (predicted_labels[pyg_data.train_mask] == pyg_data.y[pyg_data.train_mask]).sum()
            accuracy = correct_predictions / pyg_data.train_mask.sum()


        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Accuracy: {accuracy:.4f}')

    print("--- GNN 训练完成 ---")

    # 在测试集上评估最终模型性能
    model.eval()
    with torch.no_grad():
        test_out = model(pyg_data.x, pyg_data.edge_index)
        # 你需要定义你的测试集掩码 pyg_data.test_mask
        # test_loss = criterion(test_out[pyg_data.test_mask], pyg_data.y[pyg_data.test_mask])
        # test_predicted_labels = test_out.argmax(dim=1)
        # test_correct_predictions = (test_predicted_labels[pyg_data.test_mask] == pyg_data.y[pyg_data.test_mask]).sum()
        # test_accuracy = test_correct_predictions / pyg_data.test_mask.sum()
        # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return model


def run(g, pyg_data):
    # 你需要定义你的 `num_classes`，这取决于你的节点分类任务
    # 例如，如果你的节点类型是 'text_chunk', 'image', 'page'，那么 num_classes=3
    # 在实际应用中，你需要从你的业务逻辑中获取这些标签
    num_classes = 3  # 假设我们有 3 种节点类型需要分类

    # 在调用 train_gnn 之前，确保 pyg_data.y 和掩码 (train_mask, val_mask, test_mask) 是正确设置的
    # 这部分是 GNN 训练的关键，需要你根据实际任务来定义
    # 可以在 train_gnn 内部处理，也可以在外部准备好再传入

    # 简单示例，将所有节点都用于训练，并随机分配标签，实际项目中需要根据业务逻辑定义
    # (这只是为了让代码跑起来，实际中你需要为你的节点提供有意义的标签)
    all_node_types = [g.nodes[pyg_data.idx_to_node_id[i]]['type'] for i in range(pyg_data.num_nodes)]
    # 简单的类型到索引映射
    type_to_label_map = {'text_chunk': 0, 'image': 1, 'page': 2}
    # 检查是否有未知的类型
    for node_type in all_node_types:
        if node_type not in type_to_label_map:
            print(
                f"Warning: Unknown node type '{node_type}' encountered. Please update type_to_label_map and num_classes.")
            # 动态添加或处理未知类型
            type_to_label_map[node_type] = len(type_to_label_map)

    num_classes = len(type_to_label_map)  # 更新类别数量

    labels = [type_to_label_map[node_type] for node_type in all_node_types]
    pyg_data.y = torch.tensor(labels, dtype=torch.long)

    # 简单的训练/验证/测试划分 (你需要根据你的数据集和任务来调整)
    # 比如，你可以按页面划分，或者按节点类型划分
    num_nodes = pyg_data.num_nodes
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 随机打乱索引
    indices = torch.randperm(num_nodes)
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))

    pyg_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pyg_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pyg_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    pyg_data.train_mask[indices[:train_end]] = True
    pyg_data.val_mask[indices[train_end:val_end]] = True
    pyg_data.test_mask[indices[val_end:]] = True

    trained_model = train_gnn(g, pyg_data, num_classes)

    print("\nGNN 运行完毕！你可以使用 trained_model 进行预测或进一步分析。")
