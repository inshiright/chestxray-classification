# Overview
This project compares Generic CV Models (ImageNet-based) and Domain-Specific Foundation Models (Medical-based) for automated multi-label classification of thoracic pathologies in chest X-ray imaging.

## Requirements:
- Python version of 3.11 and above is used for this project.
- CUDA is required for this project. Open a terminal from the root folder and run `python scripts/cuda_check.py`.
<!-- Alternative phrasing: -->
<!-- Run `cuda_check.py` under the `scripts` folder to ensure CUDA is available. -->
> Note: The python launcher utility differs from device to device. if `python` does not work, try using `py` or `py3`. Example: `py3 scripts/cuda_check.py`

## Downloading Dependencies:
Open a terminal from the root folder and run `python -m pip install -r requirements.txt`.
> Note: The python launcher utility differs from device to device. if `python` does not work, try using `py` or `py3`. Example: `py3 -m pip install -r requirements.txt`

## Dataset Setup:
This project uses the NIH Chest X-rays dataset.  
Due to size limitations, the dataset is not included in the repository.

### Step 1 — Generate Kaggle API token
1. Log in/Sign up to an account at [Kaggle](https://www.kaggle.com/).
2. Click on your profile picture (top right) and select Settings.
3. Scroll down to the API section.
4. Click 'Create Legacy API Key'. This will download a file named `kaggle.json` to your computer.
5. Open your File Explorer and go to `C:\Users\<YourUsername>`.
6. Create a new folder named `.kaggle` (if it doesn't exist).
7. Move the `kaggle.json` file from your Downloads folder into the `.kaggle` folder.

### Step 2 — Download dataset
Open a terminal from the root folder and run `python scripts/download_dataset.py`.  
The dataset will be downloaded automatically.  
> Note: The python launcher utility differs from device to device. if `python` does not work, try using `py` or `py3`. Example: `py3 scripts/download_dataset.py`