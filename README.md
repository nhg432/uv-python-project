# OpenOrganelle Data Explorer

## Overview
This project provides tools for exploring and downloading cellular imaging data from the OpenOrganelle platform. It includes a Python downloader module and an interactive Jupyter notebook for data visualization and analysis.

## Features
- 🔬 Access to OpenOrganelle FIB-SEM datasets
- 📊 Interactive data exploration with Jupyter notebooks
- 🎨 Cellular imaging data visualization
- 📥 Efficient data downloading and caching
- 🧮 Integration with zarr, dask, and other scientific computing tools

### Quick start: Jupyter Notebook

Open and run the interactive notebook:
```bash
jupyter notebook openorganelle_explorer.ipynb
```

The notebook provides:
- Environment verification
- Dataset exploration 
- Sample data download
- Data visualization
- Advanced data access patterns

## Project Structure
```
uv-python-project/
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── openorganelle_downloader.py    # Core downloader module
├── data/                              # Downloaded data (gitignored)
├── downloads/                         # Download cache (gitignored)
├── openorganelle_explorer.ipynb       # Interactive exploration notebook
├── pyproject.toml                     # UV project configuration
├── requirements.txt                   # Python dependencies
├── uv.lock                           # UV lock file
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+ 
- [uv](https://docs.astral.sh/uv/) package manager

### Setup
1. Clone the repository:
```bash
git clone <your-repo-url>
cd uv-python-project
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux  
source .venv/bin/activate
```

## Usage

### Command Line Interface
```bash
# List available datasets
python src/openorganelle_downloader.py --list-datasets

# Explore a specific dataset
python src/openorganelle_downloader.py --explore jrc_hela-2

# Download sample data
python src/openorganelle_downloader.py --download jrc_hela-2
```

### Jupyter Notebook
Open and run the interactive notebook:
```bash
jupyter notebook openorganelle_explorer.ipynb
```

The notebook provides:
- Environment verification
- Dataset exploration 
- Sample data download
- Data visualization
- Advanced data access patterns

## Data Sources
This project accesses data from:
- **OpenOrganelle**: https://openorganelle.janelia.org/
- **Janelia COSEM**: High-resolution cellular imaging datasets
- **FIB-SEM volumes**: Focused Ion Beam Scanning Electron Microscopy data
- **Organelle segmentations**: Machine learning-generated cellular component labels

## Dependencies
Key scientific computing packages:
- `numpy` - Numerical computing
- `matplotlib` - Data visualization  
- `zarr` - Chunked array storage
- `dask` - Parallel computing
- `fsspec` - File system interfaces
- `tqdm` - Progress bars
- `jupyter` - Interactive notebooks

## Examples
See the `openorganelle_explorer.ipynb` notebook for detailed examples of:
- Connecting to OpenOrganelle datasets
- Downloading and caching sample data
- Creating multi-dimensional visualizations
- Accessing large datasets with dask arrays
- Working with organelle segmentation data

## Development

### Adding New Features
1. Create a new branch: `git checkout -b feature-name`
2. Make changes and test thoroughly
3. Update documentation as needed
4. Submit a pull request

### Testing
```bash
# Run the main module
python src/main.py

# Test the downloader
python src/openorganelle_downloader.py --help
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

Areas for contribution:
- Additional visualization methods
- Support for more data formats
- Performance optimizations
- Documentation improvements

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Resources
- **OpenOrganelle Documentation**: https://github.com/janelia-cosem/fibsem-tools
- **CellMap Project**: https://www.janelia.org/project-team/cellmap
- **UV Documentation**: https://docs.astral.sh/uv/
- **Zarr Documentation**: https://zarr.readthedocs.io/