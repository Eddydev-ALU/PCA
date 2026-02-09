# Principal Component Analysis (PCA) from Scratch

## Overview

This project implements Principal Component Analysis (PCA) from scratch using NumPy, without relying on sklearn's PCA implementation. The implementation demonstrates dimensionality reduction techniques on the CO2 Emission Africa dataset, which contains comprehensive environmental and economic data across African countries.

## Features

- **Task 1**: Complete PCA implementation from scratch (standardization, covariance matrix, eigendecomposition, projection)
- **Task 2**: Dynamic component selection based on explained variance
- **Task 3**: Performance optimization for large-scale datasets

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Eddydev-ALU/PCA.git
```

### 2. Create a Virtual Environment (Recommended)

**Important**: Do NOT push the `.venv/` folder to GitHub. It's already in `.gitignore`.

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Required Dependencies

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Or install from requirements file if available:

```bash
pip install -r requirements.txt
```

### Required Python Libraries

- **numpy**: For numerical computations and matrix operations
- **pandas**: For data loading and manipulation
- **matplotlib**: For creating visualizations
- **seaborn**: For enhanced statistical plots
- **jupyter**: For running the notebook

## Dataset

The project uses `co2 Emission Africa.csv`, which contains:
- CO2 emissions data across African countries over multiple years
- Economic indicators: GDP per capita, GDP PPP
- Population and geographic data
- Emission sources: Transportation, Manufacturing/Construction, Energy, Electricity/Heat
- Land-use and forestry emissions
- Industrial processes and fugitive emissions
- Building and bunker fuels emissions
- Multiple years of data (2000 onwards)
- Rich multivariate data with 15+ numerical features after preprocessing

## Usage

### Running the Notebook

1. **Start Jupyter Notebook**

```bash
jupyter notebook
```

2. **Open the Notebook**

Navigate to and open: `PCA_Formative_1_Eddy.ipynb`

3. **Run All Cells**

- Click "Kernel" → "Restart & Run All"
- Or run cells sequentially using Shift+Enter

### Using the PCA Implementation

The notebook is organized into clear steps:

#### Step 1: Load and Standardize Data
```python
# Loads the dataset and performs manual standardization
# Formula: (X - mean) / std_dev
standardized_data = (data - data_mean) / data_std
```

#### Step 2: Calculate Covariance Matrix
```python
# Computes covariance matrix manually
cov_matrix = (1 / (n_samples - 1)) * np.dot(standardized_data.T, standardized_data)
```

#### Step 3: Eigendecomposition
```python
# Extracts eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

#### Step 4: Sort and Select Components
```python
# Sorts by explained variance and selects components
sorted_indices = np.argsort(eigenvalues)[::-1]
```

#### Step 5: Project Data
```python
# Projects data onto selected principal components
reduced_data = np.dot(standardized_data, selected_eigenvectors)
```

### Customization

**Adjust Variance Threshold:**

In the notebook, modify the variance threshold to retain different amounts of information:

```python
# Change this value (default: 0.90 = 90%)
variance_threshold = 0.90  # Adjust between 0.80 - 0.99
```

**Select Specific Number of Components:**

```python
# Manually set number of components
num_components = 2  # Choose any number <= total features
```

## Expected Output

### Visualizations

1. **Covariance Matrix Heatmap**: Shows correlations between features
2. **Explained Variance Charts**: Bar chart and cumulative variance plot
3. **Before/After PCA Comparison**: 2D scatter plots
4. **3D Visualization**: Original data in 3D space
5. **PCA Biplot**: Feature contributions to principal components
6. **Feature Importance**: Bar chart showing loadings

### Performance Metrics

- Dimension reduction percentage
- Variance retained
- Memory savings
- Processing time and throughput
- Scalability estimates

### Console Output

- Dataset information and shape
- Standardized data preview
- Eigenvalues and eigenvectors
- Explained variance ratios
- Reduced data statistics

## Project Structure

```
PCA/
├── PCA_Formative_1_Eddy.ipynb 
├── co2 Emission Africa.csv                           
├── README.md                                           
├── .gitignore                                         
└── .venv/                                              # Virtual environment (not in repo)
```

**Note**: The `.venv/` directory is created locally and is listed in `.gitignore`. Each user should create their own virtual environment following the installation steps above.

## Implementation Details

### Performance Optimization

- Vectorized operations throughout
- Efficient memory management
- O(n × m²) complexity for covariance
- O(m³) for eigendecomposition
- Scalable to millions of samples

## Troubleshooting

### Import Errors

```bash
# If you get "ModuleNotFoundError"
pip install --upgrade numpy pandas matplotlib seaborn
```

### Kernel Issues

```bash
# Install ipykernel in your virtual environment
pip install ipykernel
python -m ipykernel install --user --name=venv
```

### File Not Found

Ensure `co2 Emission Africa.csv` is in the same directory as the notebook.

## Results Summary

- **Dataset**: CO2 Emission Africa comprehensive environmental data
- **Features**: 15+ variables including CO2 emissions, economic indicators, population data
- **Dynamic Component Selection**: Based on 90% variance threshold
- **Performance**: Optimized for large-scale processing
- **Throughput**: Efficiently handles datasets with hundreds of samples and features

## License

Educational use only.
