#!/bin/bash

# 0. Optional: Install virtual environment support (Ubuntu/Debian)
# You can skip this if `python3 -m venv` works
sudo apt update && sudo apt install -y python3-venv

# 1. Create a virtual environment in `.venv` folder
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip 
pip install --upgrade pip

# 4. Install required Python package (TensorFlow)
pip install tensorflow

# 5. Run the Python script to extract weights
python extract_gpt2_checkpoint.py

