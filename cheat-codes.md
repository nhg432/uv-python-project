# Cheat codes

This file has helpful code snippets, for re-use.

## Basics

* hold `ctrl` when using arrows to skip over entire words instead of single letters. Hold `^` (aka `shift`) to highlight.
* press up (and down) arrows, in the terminal or console, to return to previous commands that you've executed.

## OpenOrganelle Data Downloader

**List available datasets:**
```bash
python src/openorganelle_downloader.py --list-datasets
```

**Explore a specific dataset:**
```bash
python src/openorganelle_downloader.py --explore jrc_hela-2
```

**Download sample data:**
```bash
python src/openorganelle_downloader.py --download jrc_hela-2 --sample-size 64
```

**Run the example script:**
```bash
python example_usage.py
```

**Open the Jupyter notebook:**
```bash
jupyter notebook openorganelle_explorer.ipynb
```

## Activating the uv virtual environment

**Windows PowerShell:**
```powershell
.\uv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
uv\Scripts\activate.bat
```

**Git Bash:**
```bash
source uv/Scripts/activate
```

**Deactivating the virtual environment:**
```bash
deactivate
```

