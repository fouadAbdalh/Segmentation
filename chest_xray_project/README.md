# Chest X-Ray Heart Segmentation with PyTorch U-Net

This project performs **heart segmentation** on chest X-ray images using a **U-Net model** implemented in **PyTorch**.  

---

## Features
- Multi-class segmentation of chest X-ray images
- Trains on dataset with masks for heart region
- Real-time predictions with PyTorch
- Simple interface for uploading images

---

## Dataset
- Train Images: `train_images/`  
- Test Images: `test_images/`  
- Each train folder contains:
  - `images/` → X-ray image
  - `masks/` → Ground truth mask  

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/chest-xray-segmentation.git
cd chest-xray-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

