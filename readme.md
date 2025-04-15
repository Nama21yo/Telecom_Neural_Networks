This project analyzes telecom customer churn data to predict which customers are likely to leave the service. I built two different prediction models (logistic regression and neural network) and compared their performance. The data contains customer information like call failures, SMS frequency, and subscription details collected over 9 months, with churn status after 12 months.

## What I Did

### Data Preprocessing

- Checked for missing values and handled them
- Explored features through visualization
- Split data into training (80%) and testing (20%) sets
- Applied standard scaling to normalize the features

### Model 1: Logistic Regression with scikit-learn

- Implemented logistic regression which naturally uses the sigmoid function
- Experimented with different regularization parameters (C values: 0.01, 0.1, 1, 10, 100)
- Analyzed how C affects model performance metrics
- Identified the most influential features for churn prediction

### Model 2: PyTorch Neural Network

- Built a neural network with:
  - 3 hidden layers (64, 32, 16 neurons)
  - ReLU activation for hidden layers
  - Sigmoid activation for output layer
  - Dropout (0.2) for regularization
- Used Binary Cross Entropy Loss and Adam optimizer
- Trained for 100 epochs with batch size of 32
- Monitored training and validation loss

### Handling Class Imbalance

- Analyzed the class distribution
- Implemented class weighting in logistic regression
- Compared performance metrics before and after addressing imbalance

### Model Comparison and Evaluation

- Compared models using:
  - Accuracy
  - ROC AUC score
  - Precision, recall, and F1-score for both classes
  - Confusion matrices
- Visualized the ROC curves for both models
- Analyzed feature importance
