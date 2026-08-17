import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components


def laplacian_eigenmaps(X, n_neighbors=25, n_components=2):
    """
    Laplacian Eigenmaps (binary weights, paper-faithful, stable)

    Parameters:
    X : (N, D) data
    n_neighbors : number of neighbors (increase for connectivity)
    n_components : output dimension

    Returns:
    Y : (N, n_components) embedding
    """

    N = X.shape[0]

    # ================================
    # 1. k-NN graph
    # ================================
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # ================================
    # 2. Build symmetric adjacency (binary weights)
    # ================================
    W = np.zeros((N, N))

    for i in range(N):
        for j in indices[i][1:]:  # skip self
            W[i, j] = 1.0
            W[j, i] = 1.0 

    # ================================
    # 3. CHECK CONNECTIVITY (CRITICAL)
    # ================================
    n_comp, labels = connected_components(W)
    print("Connected components:", n_comp)

    if n_comp > 1:
        print("⚠️ Graph not connected → increasing neighbors recommended")
        # You can also choose largest component:
        largest = np.argmax(np.bincount(labels))
        mask = (labels == largest)

        X = X[mask]
        W = W[mask][:, mask]
        N = X.shape[0]
        print("Using largest connected component with size:", N)

    # ================================
    # 4. Degree + Laplacian
    # ================================
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # ================================
    # 5. Solve generalized eigenproblem
    # ================================
    eigenvalues, eigenvectors = eigh(L, D)

    print("Smallest eigenvalues:", eigenvalues[:5])

    # ================================
    # 6. Take non-trivial eigenvectors
    # ================================
    Y = eigenvectors[:, 1:n_components + 1]

    return Y