# utils/neighbor_finder.py

import torch
from torch_geometric.nn import knn_graph, radius_graph

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS library not found. 'faiss' neighbor finder will not be available.")
    print("Install with: pip install faiss-gpu (or faiss-cpu)")


def find_neighbors(pos, batch, method, k, r):
    if method == 'knn':
        return find_neighbors_knn(pos, batch, k)
    elif method == 'radius':
        return find_neighbors_radius(pos, batch, r, k) 
    elif method == 'faiss':
        if not FAISS_AVAILABLE:
            print("Error: FAISS method selected but library is not available. Falling back to KNN.")
            return find_neighbors_knn(pos, batch, k)
        return find_neighbors_faiss_ann(pos, batch, k)
    else:
        raise ValueError(f"Unknown neighbor finder method: {method}")


def find_neighbors_knn(pos, batch, k):
    k_safe = min(k, pos.size(0) - 1)
    if k_safe <= 0: k_safe = 1
    return knn_graph(pos, k=k_safe, batch=batch, loop=False)


def find_neighbors_radius(pos, batch, r, k_max):
    return radius_graph(pos, r=r, batch=batch, loop=False, max_num_neighbors=k_max)


def find_neighbors_faiss_ann(pos, batch, k):
    device = pos.device
    pos_np = pos.cpu().numpy().astype(np.float32)
    batch_np = batch.cpu().numpy()
    unique_batches = np.unique(batch_np)
    
    all_edge_i = []
    all_edge_j = []

    for b_idx in unique_batches:
        mask = (batch_np == b_idx)
        indices_in_batch = np.where(mask)[0] 
        points_in_batch = pos_np[mask]
        
        num_points_in_batch = points_in_batch.shape[0]
        if num_points_in_batch <= k:
            if num_points_in_batch <= 1: continue
            edge_index_batch = knn_graph(pos[mask], k=num_points_in_batch-1, batch=None, loop=False)
            all_edge_i.append(torch.from_numpy(indices_in_batch[edge_index_batch[0].cpu().numpy()]))
            all_edge_j.append(torch.from_numpy(indices_in_batch[edge_index_batch[1].cpu().numpy()]))
            continue

        res = faiss.StandardGpuResources() 
        index_flat = faiss.IndexFlatL2(3)  
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index.add(points_in_batch)

        _, I = gpu_index.search(points_in_batch, k + 1)
        
        source_nodes_local = np.arange(num_points_in_batch).reshape(-1, 1).repeat(k, axis=1)
        neighbor_nodes_local = I[:, 1:] 

        valid_mask = neighbor_nodes_local != -1
        source_nodes_local = source_nodes_local[valid_mask]
        neighbor_nodes_local = neighbor_nodes_local[valid_mask]

        source_nodes_global = indices_in_batch[source_nodes_local]
        neighbor_nodes_global = indices_in_batch[neighbor_nodes_local]

        all_edge_i.append(torch.from_numpy(source_nodes_global))
        all_edge_j.append(torch.from_numpy(neighbor_nodes_global))

    if not all_edge_i:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.stack([
        torch.cat(all_edge_i),
        torch.cat(all_edge_j)
    ], dim=0).to(device)
    
    return edge_index