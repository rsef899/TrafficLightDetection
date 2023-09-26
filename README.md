# Traffic Sign Recognition Project

This project aims to recognize and predict traffic signs using machine learning models. It includes image preprocessing, model training, and prediction functionality. The project is divided into several Python scripts, each serving a specific purpose. This README will guide you on how to use these scripts effectively.

## Prerequisites

Before you start using this project, make sure you have the following dependencies installed:

- Python (>=3.6)
- NumPy
- pandas
- scikit-learn
- joblib
- TensorFlow (for MLP model)
- scikit-image (for image processing)
- Matplotlib (for visualization)

You can install these dependencies using `pip` if they are not already installed:

```bash
pip install numpy pandas scikit-learn joblib tensorflow scikit-image matplotlib
```

## Project Structure

- `main.py`: This script takes an input image and a trained model to predict the traffic sign in the image.

- `extract_images.py`: This script is used to preprocess and extract traffic sign images from a dataset. The extracted data is serialized into a joblib file for training.

- `IntegerToClass.py`: This module provides functions to map class integers to their corresponding labels.

- `svm.py`: This script trains a Support Vector Machine (SVM) model for traffic sign recognition using the preprocessed dataset.

- `mlp.py`: This script trains a Multi-Layer Perceptron (MLP) model for traffic sign recognition using the preprocessed dataset.

## Usage Instructions

### 1. Extract Images

Before using the prediction scripts (`main.py`, `svm.py`, or `mlp.py`), you need to extract images from your dataset and create a serialized dataset file. To do this, follow these steps:

1. Place your traffic sign images in a directory structure where each class has its own subdirectory.

2. Modify the `datadir` variable in `extract_images.py` to point to the parent folder containing all category image folders.

3. Run `extract_images.py`:

   ```bash
   python extract_images.py
   ```

   This script will resize and flatten the images, creating a serialized dataset file named `dataset.joblib`.

### 2. Train the Models (Optional)

If you want to train your own models, you can use `svm.py` and `mlp.py`. Note that this step is optional, as pretrained models are provided in the project directory.

1. To train the SVM model, run:

   ```bash
   python svm.py
   ```

   The trained model will be saved as `SVMmodel.joblib`.

2. To train the MLP model, run:

   ```bash
   python mlp.py
   ```

   The trained model will be saved as `MLPmodel.joblib`.

### 3. Predict Traffic Signs

To predict a traffic sign in an image using a pretrained model, follow these steps:

1. Modify the `path_to_image` and `path_to_model` variables in `main.py` to specify the path to the image you want to predict and the path to the pretrained model (SVM or MLP).

2. Run `main.py`:

   ```bash
   python main.py
   ```

   The script will load the image, preprocess it, make a prediction, and display the predicted traffic sign along with the image.

## Note

- Make sure you have the required Python packages installed and correctly configured before running any of the scripts.

- You can use the provided pretrained models (`SVMmodel.joblib` and `MLPmodel.joblib`) to perform predictions without the need for training.

- If you want to train your own models, you can modify the hyperparameters and configurations in `svm.py` and `mlp.py` to suit your needs.

- Ensure that your traffic sign dataset is structured as described in the `extract_images.py` section for successful dataset extraction.

Feel free to customize and extend this project for your specific requirements. Enjoy recognizing traffic signs with machine learning!
