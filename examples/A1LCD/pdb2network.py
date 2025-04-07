import numpy as np
import mdtraj as md
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
import matplotlib.pyplot as plt

# Visualize the graph (3D)
def plot_graph(G, box_a, box_b, box_c):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get node positions
    node_pos = np.array([G.nodes[i]['pos'] for i in G.nodes()])

    # Plot nodes
    ax.scatter(node_pos[:,0], node_pos[:,1], node_pos[:,2], 
            c='blue', s=20, alpha=0.6)

    # Plot edges
    for edge in G.edges():
        pos1 = G.nodes[edge[0]]['pos']
        pos2 = G.nodes[edge[1]]['pos']
        ax.plot([pos1[0], pos2[0]], 
                [pos1[1], pos2[1]], 
                [pos1[2], pos2[2]], 
                'gray', alpha=0.2, linewidth=0.5)

    ax.set_title("Chain Connectivity Network")
    ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_xlim([0, box_a/10])
    ax.set_ylim([0, box_b/10])
    ax.set_zlim([0, box_c/10])
    plt.tight_layout()
    plt.show()

# System parameters
box_a, box_b, box_c = 757.365, 757.365, 757.365  # box lengths in Å
n_chain = 200
n_monomer = 137  # beads per chain
cutoff = 5.0  # Å

# Load and verify system
traj = md.load('equi_model.pdb')
positions = traj.xyz[0]  # positions in Å
traj.unitcell_vectors = np.array([[[box_a, 0, 0], [0, box_b, 0], [0, 0, box_c]]])

# Reshape into chains
pos_by_chain = positions.reshape(n_chain, n_monomer, 3)
box_dims = np.array([box_a, box_b, box_c])

print('System verification:')
print(f'Total beads: {traj.n_atoms} (expected: {n_chain * n_monomer})')
print(f'Positions shape: {positions.shape}')

# --- Step 1: Identify the largest cluster ---
def find_largest_cluster(positions, cutoff, box_dims):
    """Find atom-level largest cluster with PBC"""
    n_atoms = len(positions)
    dist_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        delta = positions - positions[i]
        delta -= np.round(delta / box_dims) * box_dims  # PBC
        dist_matrix[i] = np.sqrt(np.sum(delta**2, axis=1))
    
    adj_matrix = (dist_matrix < cutoff).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    
    _, labels = connected_components(csgraph=csr_matrix(adj_matrix), 
                                    directed=False, 
                                    return_labels=True)
    
    largest_cluster_label = np.argmax(np.bincount(labels))
    return labels == largest_cluster_label

# Get atom mask for largest cluster
largest_cluster_mask = find_largest_cluster(positions, cutoff, box_dims)

# --- Step 2: Build chain-level graph for largest cluster ---
def build_chain_graph(pos_by_chain, largest_cluster_mask, cutoff, box_dims):
    """Build graph for chains in largest cluster only"""
    chain_mask = np.any(largest_cluster_mask.reshape(n_chain, n_monomer), axis=1)
    active_chains = np.where(chain_mask)[0]

    # Filter positions for active chains
    pos_by_chain = pos_by_chain[active_chains]
    n_active_chains = len(active_chains)
    #print("pos_by_chain.shape", pos_by_chain.shape)
    #print("pos_by_chain", pos_by_chain)
    
    # Calculate COMs (PBC-aware)
    coms = np.zeros((n_active_chains, 3))
    for i in range(n_active_chains):
        chain_pos = pos_by_chain[i]
        ref_pos = chain_pos[0]
        unwrapped = ref_pos + (chain_pos - ref_pos) - np.round((chain_pos - ref_pos)/box_dims)*box_dims
        coms[i] = np.mean(unwrapped, axis=0)
    
    # Build graph
    G = nx.Graph()
    for i in range(n_active_chains):
        #G.add_node(i, pos=coms[i])
        #pos_tuple = tuple(coms[i].tolist())
        #print('pos_tuple', pos_tuple)
        #G.add_node(str(i), pos=pos_tuple)  # Node ID as string
        G.add_node(str(i), pos=f"{coms[i][0]},{coms[i][1]},{coms[i][2]}")  # Node ID as string
        #print(i, coms[i])
    
    # Add edges between chains in largest cluster
    #for i_idx, i in enumerate(active_chains):
    for i in range(n_active_chains):
        for j in range(i+1, n_active_chains):
        #for j in active_chains[i_idx+1:]:
            # Check if any bead pairs are within cutoff
            dists = pos_by_chain[i][:, None, :] - pos_by_chain[j][None, :, :]
            dists -= np.round(dists/box_dims)*box_dims  # PBC
            min_dist = np.min(np.sqrt(np.sum(dists**2, axis=2)))
            
            if min_dist <= cutoff:
                #G.add_edge(i, j, weight=1.0/min_dist) # Optional: add weights
                #G.add_edge(i, j)
                G.add_edge(str(i), str(j))

    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

# Build graph
G = build_chain_graph(pos_by_chain, largest_cluster_mask, cutoff, box_dims)

nx.write_gml(G, "./A1LCD_200_OpenMpipi.gml")
print("Graph saved as 'A1LCD_200_OpenMpipi.gml'")

# Analyze the graph
print("\nGraph analysis:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")

# visualize the graph
#plot_graph(G, box_a, box_b, box_c)