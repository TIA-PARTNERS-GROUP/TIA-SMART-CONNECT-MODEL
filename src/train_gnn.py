import os
import logging
import toml
import mgclient
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
try:
    config = toml.load("../config.toml")
    db_config = config['database']
    model_config = config['model']
    train_config = config['training']
    log_config = config['logging']
except Exception as e:
    print(f"[FATAL] Error loading config.toml: {e}")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_config.get('log_file', 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Configuration loaded successfully.")

# ---------------------------------------------------------------------------
# Data Loading & Graph Construction
# ---------------------------------------------------------------------------
def fetch_and_prepare_graph():
    """
    Connects to Memgraph, fetches all nodes and edges, processes them,
    and converts them into a PyTorch Geometric HeteroData object.
    """
    logger.info("Connecting to Memgraph at %s:%d", db_config['host'], db_config['port'])
    try:
        conn = mgclient.connect(host=db_config['host'], port=db_config['port'])
        cursor = conn.cursor()
    except Exception as e:
        logger.error(f"Failed to connect to Memgraph: {e}")
        return None

    logger.info("Loading sentence embedding model: %s", model_config['embedding_model'])
    embed_model = SentenceTransformer(model_config['embedding_model'])
    embedding_dim = embed_model.get_sentence_embedding_dimension()

    logger.info("Fetching all relevant nodes from the database...")
    query_nodes = """
    MATCH (n)
    WHERE n:User OR n:Skill OR n:Strength OR n:Idea OR n:UserPost OR n:Subscription
       OR n:DailyActivity OR n:StrengthCategory OR n:SkillCategory OR n:Project
       OR n:BusinessSkill OR n:BusinessCategory OR n:Region
    RETURN
        id(n) AS memgraph_id,
        labels(n)[0] AS label,
        COALESCE(n.id, id(n)) AS original_id,
        CASE labels(n)[0]
            WHEN 'User' THEN COALESCE(n.name, n.username, '')
            ELSE COALESCE(n.name, n.content, n.title, '')
        END AS text
    """
    cursor.execute(query_nodes)
    nodes_df = pd.DataFrame(cursor.fetchall(), columns=["memgraph_id", "label", "original_id", "text"])
    logger.info("Fetched %d nodes across %d types", len(nodes_df), nodes_df['label'].nunique())

    graph_data = HeteroData()
    node_id_maps = {}

    # Encode nodes per label
    for label, group in nodes_df.groupby('label'):
        node_id_maps[label] = {oid: i for i, oid in enumerate(group['original_id'])}
        
        texts = group['text'].fillna('').tolist()
        if texts and label != 'User':  # Users will get different features
            embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=128)
            features = torch.tensor(embeddings, dtype=torch.float)
        else:
            # For users or empty text, create random features
            features = torch.randn((len(group), embedding_dim), dtype=torch.float)
        
        # Normalize features
        if len(group) > 1:
            scaler = StandardScaler()
            features = torch.tensor(scaler.fit_transform(features.numpy()), dtype=torch.float)
            
        graph_data[label].x = features
        logger.info("Created %s nodes with %s features", label, features.shape)

    # Add user-specific features
    try:
        query_user_features = """
        MATCH (u:User)
        RETURN 
            COALESCE(u.id, id(u)) AS user_id,
            COALESCE(u.age, 30) AS age,
            COALESCE(u.rating, 0) AS rating,
            COALESCE(u.experience, 0) AS experience
        """
        cursor.execute(query_user_features)
        user_features_df = pd.DataFrame(cursor.fetchall(), columns=["user_id", "age", "rating", "experience"])
        
        if not user_features_df.empty:
            user_scaler = StandardScaler()
            user_numeric_features = user_scaler.fit_transform(user_features_df[['age', 'rating', 'experience']].fillna(0))
            
            for i, row in user_features_df.iterrows():
                user_idx = node_id_maps['User'].get(row['user_id'])
                if user_idx is not None:
                    # Replace user features with numeric features
                    numeric_features = torch.tensor(user_numeric_features[i], dtype=torch.float)
                    # Pad to match embedding dimension
                    if len(numeric_features) < embedding_dim:
                        padded_features = torch.cat([numeric_features, torch.zeros(embedding_dim - len(numeric_features))])
                        graph_data['User'].x[user_idx] = padded_features
    except Exception as e:
        logger.warning(f"Could not fetch user features: {e}")

    # Fetch edges
    logger.info("Fetching all edges to build graph structure...")
    query_edges = """
    MATCH (source)-[r]->(target)
    RETURN
        labels(source)[0] AS source_label,
        COALESCE(source.id, id(source)) AS source_id,
        type(r) AS rel_type,
        labels(target)[0] AS target_label,
        COALESCE(target.id, id(target)) AS target_id
    """
    cursor.execute(query_edges)
    all_edges_df = pd.DataFrame(cursor.fetchall(), columns=["source_label", "source_id", "rel_type", "target_label", "target_id"])
    logger.info("Fetched %d edges across %d relation types", len(all_edges_df), all_edges_df['rel_type'].nunique())
    conn.close()

    for (src_lbl, rel_type, tgt_lbl), group in all_edges_df.groupby(['source_label', 'rel_type', 'target_label']):
        if src_lbl not in graph_data.node_types or tgt_lbl not in graph_data.node_types:
            continue
        source_map, target_map = node_id_maps[src_lbl], node_id_maps[tgt_lbl]
        source_indices = [source_map.get(oid) for oid in group['source_id']]
        target_indices = [target_map.get(oid) for oid in group['target_id']]
        valid_edges = [(s, t) for s, t in zip(source_indices, target_indices) if s is not None and t is not None]
        if not valid_edges:
            continue
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        graph_data[src_lbl, rel_type, tgt_lbl].edge_index = edge_index
        logger.info("Added edge type (%s - %s -> %s) with %d edges", src_lbl, rel_type, tgt_lbl, edge_index.shape[1])

    logger.info("Graph construction complete: %d node types, %d edge types", len(graph_data.node_types), len(graph_data.edge_types))
    return graph_data

# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels) 
            for edge_type in metadata[1]
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels) 
            for edge_type in metadata[1]
        }, aggr='mean')
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict):
        # Initial linear transformation
        x_dict = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        
        # First convolution
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # Second convolution
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        return x_dict

class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_channels, 1)
        )

    def forward(self, z_src, z_dst):
        z = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(z).squeeze(-1)

# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------
def train(model, predictor, data, optimizer, edge_types):
    model.train()
    predictor.train()
    optimizer.zero_grad()
    
    z_dict = model(data.x_dict, data.edge_index_dict)
    
    total_loss = 0
    for edge_type in edge_types:
        if hasattr(data[edge_type], 'edge_label_index') and hasattr(data[edge_type], 'edge_label'):
            src_type, _, dst_type = edge_type
            z_src = z_dict[src_type][data[edge_type].edge_label_index[0]]
            z_dst = z_dict[dst_type][data[edge_type].edge_label_index[1]]
            
            out = predictor(z_src, z_dst)
            loss = nn.BCEWithLogitsLoss()(out, data[edge_type].edge_label.float())
            total_loss += loss
    
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

@torch.no_grad()
def test(model, predictor, data, edge_types):
    model.eval()
    predictor.eval()
    
    z_dict = model(data.x_dict, data.edge_index_dict)
    
    auc_scores = {}
    for edge_type in edge_types:
        if hasattr(data[edge_type], 'edge_label_index') and hasattr(data[edge_type], 'edge_label'):
            src_type, _, dst_type = edge_type
            z_src = z_dict[src_type][data[edge_type].edge_label_index[0]]
            z_dst = z_dict[dst_type][data[edge_type].edge_label_index[1]]
            
            out = predictor(z_src, z_dst)
            pred = torch.sigmoid(out)
            
            # Check if we have both positive and negative samples
            unique_labels = torch.unique(data[edge_type].edge_label)
            if len(unique_labels) > 1:
                auc = roc_auc_score(data[edge_type].edge_label.cpu().numpy(), pred.cpu().numpy())
                auc_scores[str(edge_type)] = auc
            else:
                logger.warning(f"Skipping {edge_type} - only one class present")
    
    return auc_scores

# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load and prepare graph data
    graph_data = fetch_and_prepare_graph()
    if not graph_data or not graph_data.edge_types:
        logger.error("No graph data or edges found. Exiting.")
        exit(1)

    graph_data = ToUndirected()(graph_data)

    # Select edge types for training
    training_edge_types = [
        etype for etype in graph_data.edge_types
        if etype[0] == 'User' and etype[2] != 'User' and graph_data[etype].num_edges > 10
    ]
    
    if not training_edge_types:
        logger.error("No suitable edge types found for training. Exiting.")
        exit(1)
    
    logger.info("Selected %d edge types for training: %s", len(training_edge_types), training_edge_types)

    # Split the data with reduced negative sampling ratio
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=0.5,  # Reduced to handle limited negative samples
        edge_types=training_edge_types
    )
    
    train_data, val_data, test_data = transform(graph_data)
    
    # Initialize model and optimizer
    hidden_channels = 128
    out_channels = 64
    
    model = HeteroGNN(hidden_channels, out_channels, train_data.metadata())
    predictor = LinkPredictor(out_channels)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.01,
        weight_decay=1e-5
    )
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(1, 201):
        loss = train(model, predictor, train_data, optimizer, training_edge_types)
        
        if epoch % 5 == 0:
            train_auc = test(model, predictor, train_data, training_edge_types)
            val_auc = test(model, predictor, val_data, training_edge_types)
            
            avg_train_auc = sum(train_auc.values()) / len(train_auc) if train_auc else 0
            avg_val_auc = sum(val_auc.values()) / len(val_auc) if val_auc else 0
            
            logger.info(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
            logger.info(f"Train AUC: {avg_train_auc:.4f} | Types: {len(train_auc)}")
            logger.info(f"Val AUC: {avg_val_auc:.4f} | Types: {len(val_auc)}")
            
            if avg_val_auc > best_val_auc:
                best_val_auc = avg_val_auc
                patience_counter = 0
                torch.save({
                    'model': model.state_dict(),
                    'predictor': predictor.state_dict()
                }, 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model and test
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model'])
    predictor.load_state_dict(checkpoint['predictor'])
    
    test_auc = test(model, predictor, test_data, training_edge_types)
    avg_test_auc = sum(test_auc.values()) / len(test_auc) if test_auc else 0
    logger.info(f"Final Test AUC: {avg_test_auc:.4f} | Breakdown: {test_auc}")
