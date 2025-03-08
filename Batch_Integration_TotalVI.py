import scanpy as sc
import numpy as np
import scvi
import matplotlib.pyplot as plt

# Step 1: Import Required Libraries
"""
This script demonstrates the process of training and evaluating a TotalVI model for multi-batch integration and analysis using the COVID-19 dataset.
Libraries required: 
- scanpy: For preprocessing and visualization of single-cell RNA sequencing data
- numpy: For numerical operations
- scvi: For using the scVI and TotalVI models
- matplotlib: For plotting training and validation loss
"""

# Step 2: Load the Dataset
"""
Load the dataset into an AnnData object. The dataset used here is a subset of the COVID-19 dataset.
Make sure the dataset path is correct and points to the .h5ad file.
"""
adata = sc.read_h5ad("/kaggle/input/covid-ai/covid_subsampled.h5ad")

# Step 3: Preprocess the Data
"""
Preprocess the dataset by filtering out cells and genes based on quality control metrics.
The dataset is then split into training, validation, and test sets. 
Data normalization and log-transformation are performed as part of preprocessing.
"""
# Quality control: Filter cells and genes based on specific criteria
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalize and log-transform the data
np.random.seed(42)
adata.obs['split'] = np.random.choice(['train', 'val', 'test'], size=adata.n_obs, p=[0.7, 0.15, 0.15])

# Split the dataset into train, validation, and test sets
train_adata = adata[adata.obs['split'] == 'train'].copy()
val_adata = adata[adata.obs['split'] == 'val'].copy()
test_adata = adata[adata.obs['split'] == 'test'].copy()

# Display the sizes of the subsets
print(f"Train set size: {train_adata.n_obs} cells")
print(f"Validation set size: {val_adata.n_obs} cells")
print(f"Test set size: {test_adata.n_obs} cells")

# Step 4: Prepare the Dataset for TotalVI
"""
Here, we prepare the dataset for TotalVI. In this example, synthetic protein expression data is generated.
In a real-world scenario, the protein expression data would be part of the input dataset.
"""
# Generate synthetic protein expression data for the demonstration
protein_expression_data = np.random.poisson(lam=10, size=(train_adata.n_obs, 10)).astype(int)
train_adata.obsm["protein_expression"] = protein_expression_data
test_adata.obsm["protein_expression"] = np.random.rand(test_adata.n_obs, 10)  # similarly for test set

# Set up the AnnData object for TotalVI
scvi.model.TOTALVI.setup_anndata(
    train_adata,
    batch_key="batch",  # Ensure batch key is assigned
    protein_expression_obsm_key="protein_expression"  # Protein expression data key
)

# Step 5: Initialize and Train the TotalVI Model
"""
Initialize the TotalVI model with specific parameters. We use a latent space of 10 dimensions and set the 
gene and protein dispersion types. The model is trained for up to 1000 epochs with early stopping based on validation loss.
"""
# Initialize the TotalVI model
model_totalvi = scvi.model.TOTALVI(
    train_adata,
    n_latent=10,  # Latent space dimensions
    gene_dispersion="gene",  # Gene dispersion type
    protein_dispersion="protein",  # Protein dispersion type
    gene_likelihood="nb",  # Gene likelihood (negative binomial)
    latent_distribution="normal",  # Latent space distribution
    empirical_protein_background_prior=True,  # Empirical protein background prior
)

# Train the model
model_totalvi.train(
    max_epochs=1000,
    early_stopping=True,
    early_stopping_monitor="elbo_validation",  # Monitor ELBO for early stopping
    early_stopping_patience=10,  # Stop after 10 epochs with no improvement
    plan_kwargs={"reduce_lr_on_plateau": True},  # Reduce learning rate on plateau
)

# Step 6: Plot the Training and Validation Loss
"""
Plot the training and validation loss (Negative ELBO) to evaluate model convergence. This plot helps visualize 
how well the model fits the data and provides insights into the early stopping mechanism.
"""
plt.figure(figsize=(8, 6))
plt.plot(model_totalvi.history["elbo_train"], label="Train Loss", color="blue")
plt.plot(model_totalvi.history["elbo_validation"], label="Validation Loss", color="orange")
plt.title("Negative ELBO Over Training Epochs")
plt.xlabel("Epochs")
plt.ylabel("Negative ELBO")
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss_plot.png")  # Save the plot
plt.show()

# Step 7: Setup and Test the TotalVI Model on the Test Set
"""
After training the model on the training set, we apply the trained model to the test set to obtain latent representations.
We also perform UMAP embedding on the test set to visualize the results, colored by batch or cell type.
"""
# Setup the test data for TotalVI
scvi.model.TOTALVI.setup_anndata(
    test_adata,
    batch_key="batch",
    protein_expression_obsm_key="protein_expression"
)

# Get the latent representation for the test data
test_latent_representation = model_totalvi.get_latent_representation(test_adata)
test_adata.obsm["X_totalVI"] = test_latent_representation

# Compute the UMAP embedding for the test set
sc.pp.neighbors(test_adata, use_rep="X_totalVI")
sc.tl.umap(test_adata)

# Visualize UMAP embeddings with batch or cell type colorings
sc.pl.umap(test_adata, color=["batch", "celltype"], save="_test_umap.png")