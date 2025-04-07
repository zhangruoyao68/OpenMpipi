import numpy as np
import mdtraj as md
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# System parameters
box_a, box_b, box_c = 757.365, 757.365, 757.365  # box lengths in Å
n_chain = 200
n_monomer = 137  # beads per chain
cutoff = 5.0  # Å

# Load and verify system
traj = md.load('equi_model.pdb')
positions = traj.xyz[0]  # positions in Å
traj.unitcell_vectors = np.array([[[box_a, 0, 0], [0, box_b, 0], [0, 0, box_c]]])

print('System verification:')
print(f'Total beads: {traj.n_atoms} (expected: {n_chain * n_monomer})')
print(f'Positions shape: {positions.shape}')

def find_largest_cluster(positions, cutoff, box_dims):
    """Improved cluster detection with PBC"""
    n_atoms = len(positions)
    dist_matrix = np.zeros((n_atoms, n_atoms))
    
    # Compute distances with PBC
    for i in range(n_atoms):
        delta = positions - positions[i]
        delta -= np.round(delta / box_dims) * box_dims  # minimum image
        dist_matrix[i] = np.sqrt(np.sum(delta**2, axis=1))
    
    # Build adjacency matrix
    adj_matrix = (dist_matrix < cutoff).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    
    # Find connected components
    n_components, labels = connected_components(
        csgraph=csr_matrix(adj_matrix), 
        directed=False, 
        return_labels=True
    )
    
    # Analyze clusters
    cluster_sizes = np.bincount(labels)
    largest_idx = np.argmax(cluster_sizes)
    return n_components, cluster_sizes[largest_idx], labels

# Run analysis
box_dims = np.array([box_a, box_b, box_c])
n_clusters, largest_size, cluster_labels = find_largest_cluster(positions, cutoff, box_dims)

# Calculate chain participation
chain_participation = np.zeros(n_chain)
for atom_idx, label in enumerate(cluster_labels):
    if label == np.argmax(np.bincount(cluster_labels)):  # largest cluster
        chain_id = atom_idx // n_monomer
        chain_participation[chain_id] = 1

print('\nCluster analysis results:')
print(f'Number of clusters: {n_clusters}')
print(f'Largest cluster size: {largest_size} beads ({largest_size/n_monomer:.1f} chains)')
print(f'Fraction of system clustered: {largest_size/len(positions):.3f}')
print(f'Chains in largest cluster: {int(np.sum(chain_participation))}')

# Optional: Save cluster information
np.savez('cluster_data.npz',
         labels=cluster_labels,
         chain_participation=chain_participation)