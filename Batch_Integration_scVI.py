# Step 1: Import Required Libraries
"""
This script demonstrates the use of scVI to analyze single-cell RNA-seq data.
The necessary libraries are imported at the beginning: scanpy, numpy, scvi, and matplotlib.
"""

import scanpy as sc
import numpy as np
import scvi
import matplotlib.pyplot as plt


# Step 2: Load Dataset
"""
In this step, we load the dataset into an AnnData object. 
The dataset used here is a subset of the COVID-19 dataset.
Replace the path with your dataset if necessary.
"""
# Replace the path with the correct dataset location
adata = sc.read_h5ad("/kaggle/input/covid-ai/covid_subsampled.h5ad")


# Step 3: Preprocess the Data
"""
Here, we perform data preprocessing by filtering cells and genes based on quality control metrics.
After that, we normalize the data and apply a log-transformation.
"""
# Filter cells and genes based on quality control
sc.pp.filter_cells(adata, min_genes=200)  # Keep cells with at least 200 genes
sc.pp.filter_genes(adata, min_cells=3)    # Keep genes present in at least 3 cells

# Normalize and log-transform the data
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize the data by total count
sc.pp.log1p(adata)  # Log-transform the data (log(x+1))

# Display basic statistics after preprocessing
print(f"Number of genes: {adata.n_vars}")
print(f"Number of cells: {adata.n_obs}")


# Step 4: Split Data into Train, Validation, and Test Sets
"""
In this step, we randomly split the dataset into training, validation, and test sets. 
We use a 70% - 15% - 15% split, ensuring reproducibility by setting a random seed.
"""
# Set random seed for reproducibility
np.random.seed(42)

# Add a column to `adata.obs` for dataset split
adata.obs['split'] = np.random.choice(['train', 'val', 'test'], size=adata.n_obs, p=[0.7, 0.15, 0.15])

# Create AnnData objects for each split (train, validation, and test)
train_adata = adata[adata.obs['split'] == 'train'].copy()
val_adata = adata[adata.obs['split'] == 'val'].copy()
test_adata = adata[adata.obs['split'] == 'test'].copy()

# Print the sizes of each dataset split
print(f"Train set size: {train_adata.n_obs} cells")
print(f"Validation set size: {val_adata.n_obs} cells")
print(f"Test set size: {test_adata.n_obs} cells")


# Step 5: Setup AnnData for scVI
"""
Before training the scVI model, we need to set up the AnnData object for scVI.
This includes registering the batch key, which is needed for batch correction if applicable.
"""
# Register the dataset with scVI
# If batch correction is needed, specify the batch key (e.g., 'batch')
scvi.model.SCVI.setup_anndata(train_adata, batch_key="batch")


# Step 6: Initialize and Train the Model
"""
Now, we initialize the scVI model with the specified hyperparameters. 
In this example, we set:
- 128 hidden units
- 10 latent variables
- 2 layers in the model
- 0.1 dropout rate

The model is then trained on the training data, with early stopping and learning rate reduction on plateau.
"""
# Initialize the scVI model with the chosen parameters
model = scvi.model.SCVI(
    train_adata,
    n_hidden=128,    # Number of hidden units
    n_latent=10,     # Number of latent variables
    n_layers=2,      # Number of layers in the model
    dropout_rate=0.1,  # Dropout rate to prevent overfitting
)

# Train the model using ONLY the training data
model.train(
    max_epochs=1000,  # Maximum number of epochs
    early_stopping=True,  # Enable early stopping
    early_stopping_monitor="elbo_validation",  # Monitor ELBO for validation loss
    early_stopping_patience=10,  # Stop if validation loss doesn't improve for 10 epochs
    plan_kwargs={"reduce_lr_on_plateau": True},  # Reduce learning rate when loss plateaus
)


# Step 7: Save the Model Weights
"""
After training the model, we save the model's weights to a specified directory for future use.
"""
# Save the model's weights to a file
model.save("/save2/scvi_model.pt")


# Step 8: Visualize Training and Validation Loss
"""
In this step, we plot the training and validation loss (ELBO loss) to visualize how well the model has trained.
This is useful for checking convergence and diagnosing overfitting or underfitting.
"""
# Retrieve training history
history = model.history

# Plot training and validation loss (ELBO)
plt.figure(figsize=(8, 6))
plt.plot(history['elbo_train'], label='Train ELBO', color='blue')
plt.plot(history['elbo_validation'], label='Validation ELBO', color='orange')
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.title('Training and Validation Loss (ELBO)')
plt.legend()
plt.grid(True)
plt.show()


# Step 9: Evaluate Reconstruction Error on Test Set
"""
Finally, we evaluate the reconstruction error of the model on the test set to assess how well the model performs 
on unseen data. A lower reconstruction error indicates better model performance.
"""
# Compute reconstruction error on the test set
test_reconstruction_error = model.get_reconstruction_error(adata=test_adata)

# Print the reconstruction error
print(f"Reconstruction Error on Test Set: {test_reconstruction_error}")