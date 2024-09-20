# Salma Chaaban - 301216551
# COMP257 - Assignment1

import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

'''
Exercise 1
'''

# 1. load the MNIST dataset
path = 'mnist_784.arff'
data, meta = arff.loadarff(path) # meta, contains metadata, including descriptions of the data's attributes
df = pd.DataFrame(data)

# 2. Display each digit

# Decode class column
# Convert the 'class' column, which is stored as bytes, into a string format
df['class'] = df['class'].astype(str)

# Separate features (X) and labels (y)
X = df.drop('class', axis=1)  # Pixel data
y = df['class']  # Digit labels

# Display the first 10 digits
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i, ax in enumerate(axes):
    ax.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    ax.set_title(f"{y.iloc[i]}") # Set the title
    ax.axis('off')  # Hide axis
plt.show()

# 3. Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio

pca = PCA(n_components=2) # first 2 principal components
X_pca = pca.fit_transform(X)

# Retrieve the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Output the explained variance ratio of the first two principal components
print(f"Explained variance ratio of the first principal component: {explained_variance_ratio[0]}")
print(f"Explained variance ratio of the second principal component: {explained_variance_ratio[1]}")

# Plot the projections of the 1st and 2nd principal component onto a 1D hyperplane
# First Principal Component (1D projection)
first_component = X_pca[:, 0]

# Second Principal Component (1D projection)
second_component = X_pca[:, 1]

# Plot the first component on a 1D hyperplane
plt.figure(figsize=(8, 2))
plt.plot(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), 'o', alpha=0.5)
plt.title('First Principal Component Projection')
plt.xlabel('z1')
plt.show()

# Plot the second principal component on a 1D hyperplane
plt.figure(figsize=(8, 2))
plt.plot(X_pca[:, 1], np.zeros_like(X_pca[:, 1]), 'o', alpha=0.5)
plt.title('Second Principal Component Projection')
plt.xlabel('z1')
plt.show()

# 5. Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions

n_components = 154
ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
X_ipca = ipca.fit_transform(X)

# 6. Display the original and compressed digits from
# Function to plot compressed and original images side by side
def plot_compressed_vs_original(original, compressed):
    fig, axes = plt.subplots(1, 2, figsize=(2, 1))
    
    # Original image
    axes[0].imshow(original.reshape(28, 28), cmap='gray', interpolation="nearest")
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Compressed and then inverse transformed image
    reconstructed = ipca.inverse_transform(compressed)
    axes[1].imshow(reconstructed.reshape(28, 28), cmap='gray', interpolation="nearest")
    axes[1].set_title("Compressed")
    axes[1].axis('off')
    
    plt.show()

# Display original and compressed digits
for i in range(10):
    plot_compressed_vs_original(X.iloc[i].values, X_ipca[i])



'''
Exercise 2
'''
# 1. Generate Swiss roll dataset
X_swiss, color = make_swiss_roll(n_samples=1500, noise=0.2, random_state=51)

# Plot the resulting generated Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=color, s=15)
ax.set_title("Swiss Roll Dataset")
ax.view_init(azim=-66, elev=12) == ax.text2D(0.8, 0.05, s="n_samples=1500", transform=ax.transAxes)
plt.show()

# 3. Use Kernel PCA (kPCA) with linear kernel, a RBF kernel, and a sigmoid kernel

# Linear Kernel PCA
kpca_linear = KernelPCA(n_components=2, kernel='linear')
X_kpca_linear = kpca_linear.fit_transform(X_swiss)

# RBF Kernel PCA
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_kpca_rbf = kpca_rbf.fit_transform(X_swiss)

# Sigmoid Kernel PCA
kpca_sigmoid = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.001, coef0=1)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X_swiss)

# 4. Plot the kPCA results of applying the linear kernel, a RBF kernel, and a sigmoid kernel
# Function to plot the transformed data
def plot_kpca(X_kpca, title):
    plt.figure(figsize=(4, 3))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, s=15)
    plt.title(title)
    plt.show()

# Plot the projections for different kernels
plot_kpca(X_kpca_linear, "Linear Kernel")
plot_kpca(X_kpca_rbf, "RBF Kernel")
plot_kpca(X_kpca_sigmoid, "Sigmoid Kernel")

# 5. Using kPCA and a kernel of your choice, apply Logistic Regression for classification. Use GridSearchCV
# to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end 
# of the pipeline. Print out best parameters found by GridSearchCV

labels = (color > 9).astype(int)  # Binary labels

# Pipeline with KPCA and Log Reg
clf = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('logreg', LogisticRegression())
])

# parameter grid for GridSearchCV
param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
    "kpca__gamma": np.linspace(0.03, 0.05, 20)
}

# apply GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_swiss, labels)

print(f"Best parameters found: {grid_search.best_params_}")

# 6. Plot the results from using GridSearchCV
X_kpca = grid_search.best_estimator_["kpca"].transform(X_swiss)

plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color)
plt.show()

