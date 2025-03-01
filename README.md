# heart_disease_prediction

This project aims to develop a predictive model for heart disease diagnosis using various health metrics from heart patients. The dataset includes information such as age, blood pressure, heart rate, and other relevant features, with the goal of accurately identifying individuals at risk of heart disease. Given the significant consequences of missing a positive diagnosis, the project emphasizes high recall for the positive class.

Overview
In this project, we will:

Load and explore the dataset to understand its structure, characteristics, and relevant features.
Perform exploratory data analysis (EDA) to visualize and analyze the relationships between variables, focusing on the most relevant ones for heart disease prediction.
Preprocess the data, including handling missing values, removing irrelevant features, encoding categorical variables, and scaling the features to make them suitable for machine learning models.
Build and evaluate multiple machine learning models, including Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM). We'll fine-tune their hyperparameters to improve performance, with a strong emphasis on maximizing recall to ensure we capture as many positive cases (patients with heart disease) as possible.

Requirements:

Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

Models Built:

Decision Tree Classifier: A model based on decision tree algorithms to classify the data.
Random Forest Classifier: An ensemble method that improves decision tree performance by averaging multiple trees.
K-Nearest Neighbors (KNN): A simple, non-parametric classifier based on the closest training examples in the feature space.
Support Vector Machine (SVM): A classifier that finds the optimal hyperplane separating different classes in the feature space.

Conclusion:

At the end of this project, we will evaluate the performance of each model and select the one that best balances sensitivity and specificity, ensuring we maximize recall for the positive class (heart disease).
