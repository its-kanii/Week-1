# Waste Classification using CNN Model

## Overview
This project implements a Convolutional Neural Network (CNN) model to classify images of waste into different categories. The goal is to leverage deep learning techniques to automate waste classification for better waste management.

---

## Prerequisites
Before running the code, ensure you have the following installed:

1. **Python 3.7+**
2. Libraries:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `opencv-python`
   - `tensorflow`
   - `tqdm`

You can install these dependencies using:
```bash
pip install numpy pandas matplotlib opencv-python tensorflow tqdm
```

3. Dataset:
   - The dataset should be organized into `TRAIN` and `TEST` folders, with each category of waste having its own subfolder. The paths to these folders should be updated in the code (`train_path` and `test_path`).
   - **Dataset Link**: [Download Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)

---

## File Structure
- **TRAIN Folder**: Contains subfolders for each waste category with training images.
- **TEST Folder**: Contains subfolders for each waste category with testing images.
- **Code File**: The notebook contains the implementation of the CNN model and data preprocessing steps.

---

## Key Features
1. **Data Preprocessing**:
   - Reads images from the dataset.
   - Converts images to RGB format.
   - Labels images based on their category.

2. **Model Architecture**:
   - Utilizes `Conv2D`, `MaxPooling2D`, `BatchNormalization`, and other layers to build a CNN model.
   - Supports data augmentation for improved generalization.

3. **Evaluation**:
   - Trains the model on the training dataset and evaluates it on the testing dataset.

---

## How to Run the Code
1. Download the dataset and organize it into `TRAIN` and `TEST` folders as described.
2. Update the `train_path` and `test_path` variables in the code with the paths to your dataset.
3. Run the Jupyter Notebook step by step.

---

## Outputs
The notebook provides:
- **Data Visualizations**: Plots to visualize the distribution of data.
- **Model Summary**: A detailed summary of the CNN architecture.
- **Performance Metrics**: Metrics such as accuracy and loss for training and testing data.

---

## Example Dataset Structure
```
DATASET/
├── TRAIN/
│   ├── category1/
│   ├── category2/
│   └── ...
└── TEST/
    ├── category1/
    ├── category2/
    └── ...
```

---

## Notes
- Ensure the dataset contains sufficient images in each category for effective training.
- Adjust hyperparameters like learning rate, batch size, and epochs for better results.
- Use a GPU for faster training.

---

## References
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)

---

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.

---

