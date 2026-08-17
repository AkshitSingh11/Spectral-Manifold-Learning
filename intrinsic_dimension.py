import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_intrinsic_dimension(data, discard_fraction=0.1):
    """
    Estimates the intrinsic dimension of a dataset using the TWO-NN algorithm.
    
    Parameters:
    data (numpy.ndarray or pandas.DataFrame): The high-dimensional dataset (N_samples, N_features).
    discard_fraction (float): Fraction of points with highest mu to discard (default 0.1 / 10%).
    
    Returns:
    float: The estimated intrinsic dimension (d).
    """
    # Ensure data is a numpy array
    data = np.asarray(data)
    N = data.shape[0]
    
    if N < 3:
        raise ValueError("Dataset must contain at least 3 points to find a 2nd nearest neighbor.")

    # Step 1 & 2: Compute distances to the 1st and 2nd nearest neighbors
    # We use n_neighbors=3 because the 0-th neighbor is the point itself (distance = 0)
    nn = NearestNeighbors(n_neighbors=3, algorithm='auto')
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    
    # Extract r1 and r2
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Safety check: avoid division by zero if duplicate points exist (r1 = 0)
    # We add a tiny machine epsilon to r1 where it equals 0
    r1 = np.maximum(r1, np.finfo(float).eps)
    
    # Step 3: Compute mu
    mu = r2 / r1
    
    # Step 4: Compute the empirical cumulate F_emp(mu)
    # Sort mu in ascending order
    sort_indices = np.argsort(mu)
    mu_sorted = mu[sort_indices]
    
    # F_emp(mu) = i / N, where i is the rank (1 to N)
    F_emp = np.arange(1, N + 1) / N
    
    # Step 5: Fit the line, discarding the heavy tail (outliers)
    # We keep the first (1 - discard_fraction) of the points
    keep_n = int(N * (1 - discard_fraction))
    
    mu_kept = mu_sorted[:keep_n]
    F_kept = F_emp[:keep_n]
    
    # Compute the x and y coordinates for the linear fit
    x = np.log(mu_kept)
    y = -np.log(1 - F_kept)
    
    # Fit a straight line passing through the origin: y = d * x
    d, _, _, _ = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)
    
    return d[0]
