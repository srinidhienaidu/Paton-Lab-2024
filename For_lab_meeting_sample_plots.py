

# %% 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis

# Generate a sample dataset
np.random.seed(42)
data = np.random.randn(8, 4)  # 8 data points, 4 features

# Apply PPCA (using PCA as a substitute)
ppca = PCA(n_components=2)
ppca_transformed = ppca.fit_transform(data)
ppca_eigenvector = ppca.components_[0]  # First eigenvector

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2)
fa_transformed = fa.fit_transform(data)
W_matrix = fa.components_[0]  # First row of matrix W

# Project data onto the first eigenvector for PPCA
ppca_projections = np.dot(data, ppca_eigenvector)

# Project data onto the first row of W for FA
fa_projections = np.dot(data, W_matrix)

# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# PPCA Plot (Projection onto First Eigenvector)
ax1.scatter(ppca_projections, np.zeros_like(ppca_projections), color='red', s=100)
for i in range(len(ppca_projections)):
    ax1.plot([data[i, 0], ppca_projections[i]], [0, 0], 'r--', alpha=0.5)
    ax1.text(ppca_projections[i], 0, f'Data {i+1}', fontsize=12, ha='right')

# Draw PPCA eigenvector as an axis
ax1.quiver(0, 0, ppca_eigenvector[0], ppca_eigenvector[1], angles='xy', scale_units='xy', scale=1, color='green', linewidth=2, label='First Eigenvector')

ax1.set_title('Projection onto First Eigenvector (PPCA)')
ax1.set_xlabel('Projection')
ax1.set_ylabel('')

# FA Plot (Projection onto First Row of W)
ax2.scatter(fa_projections, np.zeros_like(fa_projections), color='blue', s=100)
for i in range(len(fa_projections)):
    ax2.plot([data[i, 0], fa_projections[i]], [0, 0], 'b--', alpha=0.5)
    ax2.text(fa_projections[i], 0, f'Data {i+1}', fontsize=12, ha='right')

# Draw FA matrix W row as an axis
ax2.quiver(0, 0, W_matrix[0], W_matrix[1], angles='xy', scale_units='xy', scale=1, color='purple', linewidth=2, label='First Row of W')

ax2.set_title('Projection onto First Row of W (FA)')
ax2.set_xlabel('Projection')
ax2.set_ylabel('')

plt.tight_layout()
plt.show()


# %%
