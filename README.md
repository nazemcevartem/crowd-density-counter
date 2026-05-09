# Crowd Density Counter

CNN-based crowd counting system that estimates the number of people in an image by predicting a **density map** rather than performing direct regression. The sum of all pixel values in the predicted density map equals the estimated person count.

This project implements a **CSRNet-like architecture** in TensorFlow/Keras, featuring a VGG16 backbone, dilated convolutions for expanded receptive field, and a custom combined loss function that penalizes both pixel-level density errors and overall count deviation.

---

## Overview

Instead of predicting a single number, the model learns to generate a heatmap where high-intensity regions correspond to people. This approach is especially effective for dense crowds with heavy occlusion.

**Key idea:**  
`Estimated Count = Σ(density_map) / SCALE_FACTOR`

---

## Architecture

| Component        | Details                                                    |
|------------------|------------------------------------------------------------|
| **Encoder**      | VGG16 (up to `block3_pool`) — frozen ImageNet weights      |
| **Dilated Backend** | 4× Conv2D with dilation rate=2 + BatchNorm               |
| **Decoder**      | 3× Conv2DTranspose layers (32→64→128→256)                  |
| **Output**       | 1-channel ReLU activation (density ≥ 0)                    |

---

## Features

- **Combined Loss:** `MSE(density_map) + α · MSE(person_count)`  
  Directly supervises both map quality and final count (α = 0.01).
- **Stable Training:** ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, gradient clipping (`clipnorm=1.0`), and BatchNormalization.
- **Single Scale Factor:** Global `SCALE_FACTOR = 1000` used consistently across data loading, loss computation, and inference.
- **Pre-computed Density Maps:** Uses the [ShanghaiTech with People Density Map](https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map) dataset from Kaggle (`.h5` format).

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- OpenCV
- h5py
- NumPy
- Matplotlib

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/crowd-density-counter.git
cd crowd-density-counter
pip install -r requirements.txt
2. Download Dataset
Set your Kaggle API token and run:

python
# In notebook or script:
!kaggle datasets download -d tthien/shanghaitech-with-people-density-map
!unzip -q shanghaitech-with-people-density-map.zip -d dataset
3. Train
python
from model import build_crowd_model, combined_loss, count_mae

model = build_crowd_model(input_shape=(256, 256, 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=combined_loss,
    metrics=[count_mae]
)

# Use data_generator() for train/val streams
model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=callbacks)
4. Inference
python
img = cv2.imread('crowd.jpg')
img_input = cv2.resize(img, (256, 256)) / 255.0
pred = model.predict(np.expand_dims(img_input, axis=0))[0, :, :, 0]
count = np.sum(pred) / SCALE_FACTOR
print(f"Estimated people: {count:.1f}")
Results (ShanghaiTech Part A)
Metric	Value
MAE	~190 people
MAPE	~45%
These results were obtained on limited consumer hardware. The model captures general trends in dense crowds (>100 people) but struggles with very small groups (1–5 people) and complex textured backgrounds.

Project Structure
crowd-density-counter/
├── README.md
├── crowd-density-counter.ipynb        # Main Colab notebook
Known Limitations
Small groups: Systematic errors on scenes with 1–5 people (missed detections or false positives).

Complex textures: Architectural details, shadows, and uniform surfaces (asphalt, tiles) introduce noise.

Deeper backbones: Experiments with ResNet-54 and U-Net-like skip connections failed to converge under the same resource constraints.
