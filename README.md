# Hugging Face Model Ecosystem Repo

This project contains datasets and analysis tools for exploring the AI ecosystem, particularly focusing on models available on HuggingFace.

[Link to dataset](https://huggingface.co/datasets/modelbiome/ai_ecosystem_withmodelcards).
[Link to full Hugging Face project](https://huggingface.co/modelbiome)

## Setup

### Virtual Environment

A Python virtual environment has been created with all necessary dependencies installed.

#### To activate the virtual environment:

**Option 1: Use the activation script**
```bash
./activate_env.sh
```

**Option 2: Manual activation**
```bash
source venv/bin/activate
```

#### To deactivate:
```bash
deactivate
```

### Dependencies

The following packages are installed in the virtual environment:

#### Core Data Science
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation and analysis
- `seaborn>=0.12.0` - Statistical data visualization
- `matplotlib>=3.7.0` - Plotting library

#### Jupyter and Development
- `jupyter>=1.0.0` - Jupyter notebook environment
- `ipykernel>=6.0.0` - Python kernel for Jupyter
- `notebook>=6.5.0` - Classic Jupyter notebook

#### Data Processing and Analysis
- `scikit-learn>=1.3.0` - Machine learning library
- `scipy>=1.10.0` - Scientific computing

#### Visualization
- `plotly>=5.15.0` - Interactive plotting
- `bokeh>=3.0.0` - Interactive visualization

#### AI/ML Specific
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - HuggingFace transformers library
- `datasets>=2.12.0` - HuggingFace datasets library

#### Utilities
- `json5>=0.9.0` - JSON handling
- `tqdm>=4.65.0` - Progress bars
- `requests>=2.31.0` - HTTP library
- `python-dotenv>=1.0.0` - Environment variable management

## Usage

### Running Jupyter Notebooks

1. Activate the virtual environment:
   ```bash
   ./activate_env.sh
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Or start Jupyter Lab:
   ```bash
   jupyter lab
   ```

### Project Files

- `get_expanded_dataset.ipynb` - Notebook for expanding JSON datasets into tabular format
- `ai_ecosystem_jsons.csv` - Original JSON dataset
- `ai_ecosystem_dataset copy.csv` - Expanded dataset copy
- `ai_ecosystem_withmodelcards copy.csv` - Dataset with model cards

## Notes

- The virtual environment is located in the `venv/` directory
- All dependencies are specified in `requirements.txt`
- Use the activation script for convenience when starting work on this project 
