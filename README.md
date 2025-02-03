# logisticregression

This Streamlit application performs a Logistic Regression analysis on the Titanic dataset to predict passenger survival. Here's a summary:

Data Loading and Exploration:

Loads Titanic training and test datasets.

Displays dataset shapes, previews, and missing values using a heatmap.

Data Preprocessing:

Drops rows with missing values.

Converts categorical features (Sex and Embarked) into numerical values.

Splits data into training and testing sets.

Model Training:

Uses Logistic Regression to train the model.

Evaluates the model using accuracy and a classification report.

User Interaction:

Allows users to input passenger details (e.g., Pclass, Sex, Age, etc.) via a sidebar.

Preprocesses user input to match the training data format.

Prediction:

Predicts survival based on user input and displays the result (Survived or Not Survived).

Shows prediction probabilities for both classes.

This app is a simple yet effective tool for exploring the Titanic dataset and performing survival predictions using logistic regression.

