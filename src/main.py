import torch
import random
import networkx as nx
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pyvis.network import Network
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from sklearn.metrics import roc_auc_score

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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
    'CCTV Installer': ['Manufacturer'],
    'Software Developer': ['AI Consultant', 'UI/UX Expert', 'Engineer'],
    'Creative Designer': ['Brand Strategist', 'UI/UX Expert'],
    'Researcher': ['AI Consultant', 'Engineer'],
    'Manufacturer': ['Logistics', 'Engineer'],
    'Logistics': ['Manufacturer'],
    'Brand Strategist': ['Creative Designer', 'UI/UX Expert'],
    'UI/UX Expert': ['Software Developer', 'Creative Designer'],
    'AI Consultant': ['Software Developer', 'Researcher'],
    'Legal Advisor': ['Financial Analyst'],
    'Financial Analyst': ['Legal Advisor'],
    'Engineer': ['Software Developer', 'Manufacturer', 'Researcher']
}

def prepare_enhanced_data(df, compatibility_map):
    edge_index = []
    labels = []
    features = []
    
    capability_to_idx = {cap: idx for idx, cap in enumerate(capabilities)}
    industry_to_idx = {ind: idx for idx, ind in enumerate(industries)}
    
    for _, row in df.iterrows():
        features.append([
            capability_to_idx[row['Capability']],
            industry_to_idx[row['Industry']],
            row['Annual Revenue (M USD)'] / 1000 
        ])
    
    features = torch.tensor(features, dtype=torch.float)
    
    capability_onehot = torch.eye(len(capabilities))[features[:, 0].long()]
    industry_onehot = torch.eye(len(industries))[features[:, 1].long()]
    
    compatibility_feature = torch.zeros(len(df), len(capabilities))

    for i, cap in enumerate(df['Capability']):
        for compat_cap in compatibility_map.get(cap, []):
            j = capabilities.index(compat_cap)
            compatibility_feature[i, j] = 1
    
    features = torch.cat([
        features,
        capability_onehot,
        industry_onehot,
        compatibility_feature
    ], dim=1)
    
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and row2['Capability'] in compatibility_map.get(row1['Capability'], []):
                edge_index.append([i, j])
                labels.append(1)
    
    num_positive = len(edge_index)

    possible_negatives = []
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and row2['Capability'] not in compatibility_map.get(row1['Capability'], []):
                possible_negatives.append((i, j))
    
    negative_samples = random.sample(possible_negatives, num_positive)
    for i, j in negative_samples:
        edge_index.append([i, j])
        labels.append(0)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=features,
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float),
        node_names=df['Name'].values,
        node_capabilities=df['Capability'].values
    )

data = prepare_enhanced_data(df, compatibility_map)

edge_indices = data.edge_index.t().numpy()
edge_labels = data.y.numpy()

train_idx, test_idx = train_test_split(
    range(edge_indices.shape[0]), 
    test_size=0.2,
    random_state=42,
    stratify=edge_labels  
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

class EnhancedLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

        self.edge_predictor = torch.nn.Sequential(
            Linear(out_channels * 2, out_channels),
            torch.nn.ReLU(),
            Linear(out_channels, 1)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv3(x, edge_index)

    def predict_edge(self, z, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=1)
        return self.edge_predictor(edge_features).view(-1)

in_channels = data.x.size(1)
model = EnhancedLinkPredictor(in_channels=in_channels, hidden_channels=32, out_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()

    z = model(train_data.x, train_data.edge_index)
    pred = model.predict_edge(z, train_data.edge_index)

    loss = criterion(pred, train_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        pred = model.predict_edge(z, data.edge_index)

        pred_class = torch.sigmoid(pred) > 0.5
        acc = (pred_class == data.y).sum().item() / len(data.y)

        try:
            auc = roc_auc_score(data.y.numpy(), torch.sigmoid(pred).numpy())
        except ValueError:
            auc = 0.5  

        return acc, auc

best_test_auc = 0
patience = 20
counter = 0

for epoch in range(1, 501): 
    loss = train()

    if epoch % 10 == 0:
        train_acc, train_auc = test(train_data)
        test_acc, test_auc = test(test_data)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
            f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
            f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}')

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

model.load_state_dict(torch.load('best_model.pt'))

def visualize_predictions(data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    G = nx.Graph()

    for i, name in enumerate(data.node_names):
        capability = data.node_capabilities[i]

        G.add_node(i, 
                   label=name,
                   title=f"{name}<br>Capability: {capability}",
                   group=capability,
                   size=10 + data.x[i, 2].item() * 5) 

    src, dst = data.edge_index

    for i, (s, d) in enumerate(zip(src, dst)):
        if data.y[i] == 1:
            G.add_edge(s.item(), d.item(), 
                       color='green', 
                       width=2,
                       title='Existing compatible connection')

    all_possible_edges = torch.combinations(torch.arange(len(data.node_names)), 2).t()
    pred_scores = torch.sigmoid(model.predict_edge(z, all_possible_edges))

    ux_nodes = [i for i, cap in enumerate(data.node_capabilities) if cap == "UI/UX Expert"]
    dev_nodes = [i for i, cap in enumerate(data.node_capabilities) if cap == "Software Developer"]

    special_edges_added = 0

    for s in ux_nodes:
        for d in dev_nodes:
            if s != d:
                edge_idx = (all_possible_edges[0] == s) & (all_possible_edges[1] == d)
                if edge_idx.any():
                    score = pred_scores[edge_idx].item()
                    if score > 0.5:  
                        G.add_edge(s, d, 
                                   color='blue', 
                                   width=1 + 2 * score,
                                   title=f'Predicted connection (score: {score:.2f})',
                                   dashes=[5, 5])
                        special_edges_added += 1

    print(f"Added {special_edges_added} special UI/UX to Developer predicted edges")

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
