import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("advertising_ef.csv")

#data cleaning              CLEAN UP THIS SECTION   
data.rename(columns={'Daily Time Spent on Site':'Daily_Time_Spent_on_Site'}, inplace=True)
#data.rename(columns={'Area Income':'Area_Income'}, inplace=True)
#data.rename(columns={'Daily Internet Usage':'Daily_Internet_Usage'}, inplace=True)
#data.rename(columns={'Clicked on Ad':'Clicked_on_Ad'}, inplace=True)
data['Daily_Time_Spent_on_Site'].fillna(data['Daily_Time_Spent_on_Site'].median(),inplace=True)
data['Age'].fillna(data['Age'].median(),inplace=True)
data['Area Income'].fillna(data['Area Income'].mean(),inplace=True)
data['Area Income'] = np.around(data['Area Income'],decimals=2)
data['Daily Internet Usage'].fillna(data['Daily Internet Usage'].mean(),inplace=True)
data['City'].fillna(data['City'].value_counts().index[0],inplace=True)
data['Country'].value_counts().index[0]
data['Country'].fillna(data['Country'].value_counts().index[0],inplace=True)
data['Country'] = data['Country'].astype('category')

#assigning numerical values and storing in another column
countries = data['Country'].unique()
country_dict = {country: i for i, country in enumerate(countries)}
data['Country_cat'] = data['Country'].map(country_dict)

data['IsMale'] = (data['Gender'] == 'Male')
data_no_strings = data.drop(['Ad Topic Line', 'City', 'Country', 'Gender', 'Timestamp', 'Country_cat'], axis=1)
correlation_matrix = data_no_strings.corr()
data_no_strings.to_csv('Cleaned_Advertisements.csv')


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_no_strings.drop("Clicked on Ad", axis=1), data_no_strings["Clicked on Ad"], test_size=0.2, random_state=42)

# Define logistic regression model
model = LogisticRegression()

# Train model on training set
model.fit(X_train, y_train)

# Define objective function
def objective_function(inputs):
    # Map input values to Ad feature
    X_test_copy = X_test.copy()
    X_test_copy["Daily_Time_Spent_on_Site"] = inputs[0] * X_test_copy["Daily_Time_Spent_on_Site"]
    X_test_copy["Age"] = inputs[1] * X_test_copy["Age"]
    X_test_copy["Area Income"] = inputs[2] * X_test_copy["Area Income"]
    X_test_copy["Daily Internet Usage"] = inputs[3] * X_test_copy["Daily Internet Usage"]
    X_test_copy["IsMale"] = X_test_copy["IsMale"]
    #X_test_copy["Country_cat"] = inputs[5] * X_test_copy["Country_cat"]

    # Make prediction on test set using logistic regression model
    y_pred = model.predict(X_test_copy)

    # Calculate prediction accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Define initial input variables
initial_inputs = [1, 1, 1, 1, "Daily_time_spent_on_site", "Age", "Area income", "Daily Internet Usage", "IsMale"]

# Define maximum number of iterations
max_iterations = 100

# Define maximum number of neighbors to explore
max_neighbors = 10

# Define step size for each neighbor
step_size = 0.1

# Define current best solution
best_inputs = initial_inputs
best_accuracy = objective_function(initial_inputs)

# Define iterations counter
iterations = 0

# Perform hill climbing
while iterations < max_iterations:
    # Generate random neighbor inputs
    neighbor_inputs = [best_inputs[i] + random.uniform(-step_size, step_size) for i in range(len(best_inputs))]

    # Evaluate neighbor inputs
    neighbor_accuracy = objective_function(neighbor_inputs)

    # If neighbor inputs have higher accuracy, update best inputs
    if neighbor_accuracy > best_accuracy:
        best_inputs = neighbor_inputs
        best_accuracy = neighbor_accuracy

    # Update iterations counter
    iterations += 1

    # If maximum number of neighbors have been explored, decrease step size
    if iterations % max_neighbors == 0:
        step_size /= 2

# Print best inputs and accuracy
print("Best inputs:", best_inputs)
print("Best accuracy:", best_accuracy)
