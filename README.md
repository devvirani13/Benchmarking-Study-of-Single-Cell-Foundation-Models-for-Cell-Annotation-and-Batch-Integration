# Benchmarking-Study-of-Single-Cell-Foundation-Models-for-Cell-Annotation-and-Batch-Integration Across Multiple Datasets

## 📌 Overview

This project benchmarks the performance of **Single-Cell Foundation Models (scFMs)** and deep generative models for **cell annotation** and **batch integration** in **single-cell RNA sequencing (scRNA-seq) analysis**. The study evaluates models like **scBERT, scGPT, scVI, and TotalVI** and compares them with traditional methods such as **SingleCellNet**.

## 📝 Abstract

Single-cell RNA sequencing (scRNA-seq) analysis faces challenges related to **dataset generalization** and **batch effects**, which impact biological signal interpretation and cross-dataset comparisons. This study assesses the effectiveness of **scFMs (scBERT & scGPT)** for cell annotation and deep generative models (**scVI & TotalVI**) for batch correction. 

### 🔹 Key Findings:
- **scGPT outperforms scBERT and SingleCellNet** in cell annotation, achieving the highest accuracy of **73.4%**.
- **scVI and TotalVI improve batch integration**, reducing technical noise and enhancing biological signal preservation.
- **Future improvements** include fine-tuning models for disease-specific datasets and exploring hybrid approaches.

## 🚀 Models and Methods

### 🔬 Cell Annotation
| Model        | Accuracy (%) | F1 Score | ARI  | NMI  |
|-------------|-------------|---------|------|------|
| **scGPT**   | 73.4        | 0.69    | 0.66 | 0.68 |
| **scBERT**  | 71.2        | 0.67    | 0.63 | 0.65 |
| **SingleCellNet** (Baseline) | 65.8  | 0.61  | 0.56 | 0.58 |

**Datasets Used:**
- **Training:** Cite-seq RNA dataset (bone marrow mononuclear cells)
- **Testing:** COVID-19 dataset, Healthy Lung Tissue dataset

### 🔀 Batch Integration
| Model  | Performance |
|--------|------------|
| **scVI**  | Best batch effect correction, stable clustering |
| **TotalVI** | Captures more biological variability, better for multi-modal integration |

