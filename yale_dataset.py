import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from code import estimate_intrinsic_dimension
from laplacian_eigenmaps import laplacian_eigenmaps

def load_yale_subject(folder_path, max_images=300):
    images = []

    for file in os.listdir(folder_path):
        if file.endswith(".pgm") and "Ambient" not in file:
            img_path = os.path.join(folder_path, file)

            img = Image.open(img_path)
            img = np.array(img, dtype=np.float32)

            # flatten image → vector
            images.append(img.flatten())

            if len(images) >= max_images:
                break

    return np.array(images)


subject_path = "cropped/yaleB39"

X = load_yale_subject(subject_path, max_images=400)

print("Dataset shape:", X.shape)


# ================================
# 2. NORMALIZE
# ================================
# remove brightness bias
X = X - X.mean(axis=1, keepdims=True)

# optional scaling
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


# ================================
# 3. INTRINSIC DIMENSION (2NN)
# ================================
d_est = estimate_intrinsic_dimension(X)
print(f"Estimated Intrinsic Dimension: {d_est:.3f}")

target_dim = int(round(d_est))


# ================================
# 4. LAPLACIAN EIGENMAPS
# ================================
embedding = laplacian_eigenmaps(
    X,
    n_neighbors=20,   # important (try 15–30)
    n_components=target_dim
)


# ================================
# 5. VISUALIZATION
# ================================
plt.figure(figsize=(6, 6))

# use index as color (proxy for variation)
colors = np.arange(len(embedding))

plt.scatter(embedding[:, 0], embedding[:, 1],
            c=colors, cmap='Spectral', s=10)

plt.title("Laplacian Eigenmaps - Yale Face (Single Subject)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.colorbar()
plt.tight_layout()
plt.show()