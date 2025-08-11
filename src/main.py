import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
import torch.nn as nn
import pandas as pd
import mgclient
from sentence_transformers import SentenceTransformer # New import
import numpy as np

def fetch_and_embed_graph_data(model_name='all-MiniLM-L6-v2'):
    """
    Connects to Memgraph, fetches users and content nodes (Ideas, Posts, Strengths),
    generates text embeddings for content, and creates user features by aggregation.
    """
    print(f"Loading sentence transformer model: {model_name}...")
    embed_model = SentenceTransformer(model_name)
    embedding_dim = embed_model.get_sentence_embedding_dimension()

    conn = mgclient.connect(host="127.0.0.1", port=7687)
    cursor = conn.cursor()

    # 1. Fetch all relevant nodes and their text content
    query_nodes = """
    MATCH (n) WHERE n:User OR n:Idea OR n:UserPost OR n:Strength
    RETURN id(n) AS memgraph_id, labels(n)[0] AS label, 
           n.id AS original_id, 
           // Use CASE to get text from different properties
           CASE labels(n)[0] 
             WHEN 'Idea' THEN n.content
             WHEN 'UserPost' THEN n.content
             WHEN 'Strength' THEN n.name
             ELSE '' 
           END AS text
    """
    cursor.execute(query_nodes)
    nodes_df = pd.DataFrame(cursor.fetchall(), columns=["memgraph_id", "label", "original_id", "text"])
    nodes_df['new_id'] = range(len(nodes_df))
    id_mapping = nodes_df.set_index('memgraph_id')['new_id'].to_dict()

    # 2. Fetch relationships between users and content
    query_edges = """
    MATCH (u:User)-[r]->(c)
    WHERE type(r) IN ['SUBMITTED', 'CREATED', 'HAS_STRENGTH']
    RETURN id(u) AS source, id(c) AS target
    """
    cursor.execute(query_edges)
    edges_df = pd.DataFrame(cursor.fetchall(), columns=["source", "target"])
    conn.close()

    if edges_df.empty:
        raise ValueError("No user-to-content connections found. Cannot build graph.")

    edges_df['source_new'] = edges_df['source'].map(id_mapping)
    edges_df['target_new'] = edges_df['target'].map(id_mapping)
    edge_index = torch.tensor(edges_df[['source_new', 'target_new']].values.T, dtype=torch.long)

    # 3. Generate node features (the core change)
    node_features = np.zeros((len(nodes_df), embedding_dim), dtype=np.float32)
    
    # Generate embeddings for content nodes
    content_nodes = nodes_df[nodes_df['text'] != '']
    if not content_nodes.empty:
        print("Generating embeddings for content nodes...")
        embeddings = embed_model.encode(content_nodes['text'].tolist(), show_progress_bar=True)
        node_features[content_nodes['new_id'].values] = embeddings

    # Generate features for user nodes by aggregating content embeddings
    print("Aggregating content embeddings to create user features...")
    for user_new_id, group in edges_df.groupby('source_new'):
        connected_content_ids = group['target_new'].values
        # Average the features of connected content nodes
        user_feature_vector = node_features[connected_content_ids].mean(axis=0)
        node_features[user_new_id] = user_feature_vector

    return nodes_df, torch.tensor(node_features), edge_index

class LinkPredictor(nn.Module):
    # This class remains the same, but it will now operate on richer embeddings
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, node_embeddings, edge_index):
        source_embedding = node_embeddings[edge_index[0]]
        target_embedding = node_embeddings[edge_index[1]]
        return (source_embedding * target_embedding).sum(dim=1)

def train(model, data, optimizer, loss_fn):
    # This function also remains largely the same
    model.train()
    optimizer.zero_grad()
    node_embeddings = model.encode(data.x, data.edge_index)
    positive_preds = model.decode(node_embeddings, data.edge_index)
    positive_labels = torch.ones(data.num_edges)
    negative_edges = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges)
    negative_preds = model.decode(node_embeddings, negative_edges)
    negative_labels = torch.zeros(negative_edges.size(1))
    all_preds = torch.cat([positive_preds, negative_preds])
    all_labels = torch.cat([positive_labels, negative_labels])
    loss = loss_fn(all_preds, all_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def recommend_content_for_user(user_original_id, model, nodes_df, data, top_k=5):
    """
    Recommends new content (Ideas, Posts, Strengths) to a user.
    """
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(data.x, data.edge_index)

    user_row = nodes_df[(nodes_df['original_id'] == user_original_id) & (nodes_df['label'] == 'User')]
    if user_row.empty:
        print("Target user not found.")
        return

    user_pyg_id = user_row.iloc[0]['new_id']

    # Find all content nodes the user is NOT already connected to
    all_content_pyg_ids = nodes_df[nodes_df['label'] != 'User']['new_id'].values
    connected_content_pyg_ids = data.edge_index[1, data.edge_index[0] == user_pyg_id].numpy()
    
    unconnected_content_pyg_ids = np.setdiff1d(all_content_pyg_ids, connected_content_pyg_ids)
    if len(unconnected_content_pyg_ids) == 0:
        print("User is already connected to all content!")
        return

    # Create potential edges to score
    source_nodes = torch.tensor([user_pyg_id] * len(unconnected_content_pyg_ids), dtype=torch.long)
    target_nodes = torch.tensor(unconnected_content_pyg_ids, dtype=torch.long)
    possible_edges = torch.stack([source_nodes, target_nodes])

    scores = model.decode(node_embeddings, possible_edges).sigmoid()

    k = min(top_k, len(unconnected_content_pyg_ids))
    top_scores, top_indices = torch.topk(scores, k)
    
    recommended_pyg_ids = unconnected_content_pyg_ids[top_indices.numpy()]
    
    print(f"\nTop {k} content recommendations for User ID {user_original_id}:")
    for i, pyg_id in enumerate(recommended_pyg_ids):
        rec_info = nodes_df.iloc[pyg_id]
        print(f"  - Recommend {rec_info['label']} (ID: {rec_info['original_id']}) "
              f"with text: '{rec_info['text'][:40]}...' (Score: {top_scores[i].item():.4f})")

def main():
    # 1. Fetch data and generate embeddings
    nodes_df, node_features, edge_index = fetch_and_embed_graph_data()
    
    graph_data = Data(x=node_features, edge_index=edge_index)
    print(f"\nGraph data loaded: {graph_data}")

    # 2. Define model and training components
    model = LinkPredictor(
        in_channels=graph_data.num_node_features,
        hidden_channels=128, # Increased size for richer embeddings
        out_channels=64
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    # 3. Train the model
    print("\nTraining GNN for link prediction...")
    for epoch in range(101):
        loss = train(model, graph_data, optimizer, loss_fn)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

    # 4. Get "smart connection" recommendations
    recommend_content_for_user(1, model, nodes_df, graph_data)

if __name__ == "__main__":
    main()
