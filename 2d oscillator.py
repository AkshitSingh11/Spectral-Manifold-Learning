import numpy as np
import matplotlib.pyplot as plt

from code import estimate_intrinsic_dimension
from laplacian_eigenmaps import laplacian_eigenmaps


# ================================
# 1. Generate 2D Oscillator Dataset
# ================================
def generate_2d_oscillator(N=2000):
    X_list = []
    t_vals = []

    for i in range(N):
        # random phases
        phi1 = np.random.uniform(0, 2*np.pi)
        phi2 = np.random.uniform(0, 2*np.pi)

        t = np.random.uniform(0, 10)

        # oscillator frequencies
        w1 = 1.0
        w2 = 1.5

        # positions
        x = np.cos(w1*t + phi1)
        y = np.cos(w2*t + phi2)

        # velocities
        vx = -w1 * np.sin(w1*t + phi1)
        vy = -w2 * np.sin(w2*t + phi2)

        X_list.append([x, vx, y, vy])
        t_vals.append(t)

    return np.array(X_list), np.array(t_vals)


X, t = generate_2d_oscillator(2500)

print("Dataset shape:", X.shape)


# ================================
# 2. Intrinsic Dimension
# ================================
d_est = estimate_intrinsic_dimension(X)
print(f"Estimated Intrinsic Dimension: {d_est:.3f}")


# ================================
# 3. Laplacian Eigenmaps
# ================================
embedding = laplacian_eigenmaps(
    X,
    n_neighbors=20,
    n_components=2
)


# ================================
# 4. Visualization
# ================================
plt.figure(figsize=(12, 5))

# projection
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 2], c=t, cmap='Spectral', s=5)
plt.title("2D Oscillator (x vs y)")
plt.xlabel("x")
plt.ylabel("y")

# embedding
plt.subplot(122)
plt.scatter(embedding[:, 0], embedding[:, 1], c=t, cmap='Spectral', s=5)
plt.title("Laplacian Eigenmaps")

plt.tight_layout()
plt.show()