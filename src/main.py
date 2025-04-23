import torch
import random
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from pyvis.network import Network
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


capabilities = ["CCTV Installer", "Software Developer", "Creative Designer", "Researcher", 
               "Manufacturer", "Logistics", "Brand Strategist", "UI/UX Expert", 
               "AI Consultant", "Legal Advisor", "Financial Analyst", "Engineer"]

industries = ["Tech", "Security", "Logistics", "Manufacturing", "Research", 
             "Creative", "Finance", "Legal", "Healthcare", "Construction"]

num_entries = 100

df = pd.DataFrame({
    'Name': [f'Company_{i}' for i in range(num_entries)],
    'Capability': [random.choice(capabilities) for _ in range(num_entries)],
    'Industry': [random.choice(industries) for _ in range(num_entries)],
    'Annual Revenue (M USD)': [random.randint(1, 1000) for _ in range(num_entries)]
})

compatibility_map = {
    'CCTV Installer': ['Security', 'Manufacturer'],
    'Software Developer': ['AI Consultant', 'UI/UX Expert', 'Engineer'],
    'Creative Designer': ['Brand Strategist', 'UI/UX Expert'],
    'Researcher': ['AI Consultant', 'Engineer'],
    'Manufacturer': ['Logistics', 'Engineer'],
    'Logistics': ['Manufacturer', 'Security'],
    'Brand Strategist': ['Creative Designer', 'UI/UX Expert'],
    'UI/UX Expert': ['Software Developer', 'Creative Designer'],
    'AI Consultant': ['Software Developer', 'Researcher'],
    'Legal Advisor': ['Financial Analyst'],
    'Financial Analyst': ['Legal Advisor'],
    'Engineer': ['Software Developer', 'Manufacturer', 'Researcher']
}

def prepare_data(df, compatibility_map):
    edge_index = []
    labels = []
    features = []
    
    capability_to_idx = {cap: idx for idx, cap in enumerate(capabilities)}
    industry_to_idx = {ind: idx for idx, ind in enumerate(industries)}
    node_mapping = {name: idx for idx, name in enumerate(df['Name'])}
    
    for _, row in df.iterrows():
        features.append([
            capability_to_idx[row['Capability']],
            industry_to_idx[row['Industry']],
            row['Annual Revenue (M USD)'] / 1000
        ])
    
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and row2['Capability'] in compatibility_map.get(row1['Capability'], []):
                edge_index.append([i, j])
                labels.append(1)
    
    num_positive = len(edge_index)
    existing_edges = set((i, j) for i, j in edge_index)

    while len(labels) < 2 * num_positive:
        i, j = random.sample(range(len(df)), 2)

        if (i, j) not in existing_edges and (j, i) not in existing_edges:
            edge_index.append([i, j])
            labels.append(0)
            existing_edges.add((i, j))
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float),
        node_names=df['Name'].values,
        node_capabilities=df['Capability'].values
    )

data = prepare_data(df, compatibility_map)

edge_indices = data.edge_index.t().numpy()
edge_labels = data.y.numpy()

train_idx, test_idx = train_test_split(
    range(edge_indices.shape[0]), 
    test_size=0.2,
    random_state=42
)

train_data = Data(
    x=data.x,
    edge_index=torch.tensor(edge_indices[train_idx].T, dtype=torch.long),
    y=torch.tensor(edge_labels[train_idx], dtype=torch.float),
    node_names=data.node_names,
    node_capabilities=data.node_capabilities
)

test_data = Data(
    x=data.x,
    edge_index=torch.tensor(edge_indices[test_idx].T, dtype=torch.long),
    y=torch.tensor(edge_labels[test_idx], dtype=torch.float),
    node_names=data.node_names,
    node_capabilities=data.node_capabilities
)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = LinkPredictor(in_channels=3, hidden_channels=16, out_channels=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()

    z = model(train_data.x, train_data.edge_index)
    src, dst = train_data.edge_index
    pred = (z[src] * z[dst]).sum(dim=1)

    loss = criterion(pred, train_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(data):
    model.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)
        src, dst = data.edge_index
        pred = (z[src] * z[dst]).sum(dim=1)
        pred = torch.sigmoid(pred) > 0.5
        correct = (pred == data.y).sum().item()
        
        return correct / len(data.y)

for epoch in range(1, 101):
    loss = train()

    if epoch % 10 == 0:
        train_acc = test(train_data)
        test_acc = test(test_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

def visualize_predictions(data, threshold=0.7):
    model.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)

    G = nx.Graph()

    for i, name in enumerate(data.node_names):
        G.add_node(i, label=name, title=f"{name}<br>Capability: {data.node_capabilities[i]}", group=data.node_capabilities[i])

    src, dst = data.edge_index

    for i, (s, d) in enumerate(zip(src, dst)):
        if data.y[i] == 1:
            G.add_edge(s.item(), d.item(), color='green', title='Existing connection')

    all_possible_edges = torch.combinations(torch.arange(len(data.node_names)), 2).t()

    with torch.no_grad():
        pred_scores = torch.sigmoid((z[all_possible_edges[0]] * z[all_possible_edges[1]]).sum(dim=1))

    top_k = min(20, len(pred_scores))
    top_indices = pred_scores.topk(top_k).indices

    for idx in top_indices:
        s, d = all_possible_edges[:, idx]
        if not G.has_edge(s.item(), d.item()):
            G.add_edge(s.item(), d.item(), color='red', title=f'Predicted connection (score: {pred_scores[idx]:.2f})', dashes=True)

    nt = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    nt.from_nx(G)
    nt.set_options("""
{
  "nodes": {
    "borderWidth": 2
  },
  "edges": {
    "smooth": {
      "type": "continuous"
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 200
  },
  "manipulation": {
    "enabled": true
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -80000,
      "centralGravity": 0.3,
      "springLength": 200
    },
    "minVelocity": 0.75
  },
  "configure": {
    "enabled": true,
    "filter": "nodes,edges",
    "showButton": true
  }
}
""")    
    nt.show('graph_predictions.html', notebook=False)

visualize_predictions(data)
