# PM10_prediction_SciML
## Overview

This repository contains code for training a Physics-Informed Neural Network (PINN) to predict air quality metrics using meteorological and pollutant data. The model is designed to predict PM10 concentrations based on various features, including meteorological conditions and other pollutants. The code leverages PyTorch for building the neural network, Optuna for hyperparameter optimization, and several preprocessing techniques to handle and transform the data.

## Physics-Informed Neural Networks (PINNs)

### What is a PINN?

Physics-Informed Neural Networks (PINNs) are a class of neural networks that incorporate physical laws into the learning process. Unlike traditional neural networks, which rely solely on data, PINNs leverage known physical equations to guide the training process. This approach helps in improving the model's generalization and accuracy, especially in scenarios where data might be scarce or noisy.

### Benefits of PINNs

Improved Generalization: By incorporating physical laws, PINNs can generalize better to unseen data, as they are not solely reliant on the training data.
Data Efficiency: PINNs can achieve good performance with less data by leveraging domain knowledge.
Interpretability: The integration of physical laws provides a more interpretable model, as the predictions are consistent with known scientific principles.
Robustness: PINNs are often more robust to noise in the data, as the physical constraints act as a regularizing factor.
### Workflow in the Code

* Model Definition: The PINN is defined using PyTorch, with layers for input, hidden, and output, along with dropout layers for regularization.
* Physics-Informed Loss: A custom loss function is defined that combines the traditional mean squared error with a physics-informed term. This term is calculated based on meteorological features and their known relationships.
* Training Process: The model is trained using a combination of data-driven and physics-informed loss, allowing it to learn from both the data and the underlying physical principles.
* Hyperparameter Optimization: Optuna is used to find the best set of hyperparameters for the model, ensuring optimal performance.
* Evaluation: The trained model is evaluated on a test set, and metrics such as MSE, MAE, and R² are calculated to assess its performance.
## Requirements

To run the code, you need the following Python packages:
- pip install pandas numpy torch optuna scikit-learn matplotlib seaborn

## Data

The dataset used in this project is stored in a CSV file named BIDHANAAGAR_complete_raw_data.csv. The file should be located in the specified path within your Google Drive: /content/drive/MyDrive/SciML/csvfiles/. The dataset contains various columns representing different pollutants and meteorological features.

## Preprocessing

Data Loading: The data is loaded from a CSV file using pandas.
Data Conversion: Columns are converted to numeric types, and timestamps are parsed into datetime objects.
Missing Values: Missing values are handled using KNN imputation.
Feature Engineering: New features are created based on existing ones, including interaction terms and trigonometric transformations.
Polynomial Features: Polynomial features are generated for non-meteorological data.
Normalization: Features and target variables are normalized using MinMaxScaler.
## Model

The model is a Physics-Informed Neural Network (PINN) implemented using PyTorch. It includes:

Multiple hidden layers with ReLU activation functions.
Dropout layers for regularization.
Physics-informed loss terms to incorporate domain knowledge into the learning process.
## Training

The training process involves:

Splitting the data into training, validation, and test sets.
Hyperparameter optimization using Optuna to find the best model configuration.
Training the model with early stopping based on validation loss.
## Evaluation

The model is evaluated on the test set using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

## Usage

To run the code, ensure you have the necessary data file and dependencies installed. Execute the script in a Python environment, such as Jupyter Notebook or a Python IDE. The script will preprocess the data, train the model, and output the evaluation metrics.

## Results

The script will print the best hyperparameters found during optimization and the evaluation metrics on the test set.

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please contact Nakul Choudhari at nakul.choudhari07@gmail.com.
