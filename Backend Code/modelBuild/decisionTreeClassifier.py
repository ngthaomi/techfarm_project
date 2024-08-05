import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
import joblib

# Read the CSV file
data = pd.read_csv('plant_data.csv')

# Separate the features and the label
features = data.drop('label', axis=1)
label = data['label']

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into train and test sets with 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.3, random_state=42)

# Create a decision tree classifier model
model = DecisionTreeClassifier()

# Perform 5-fold cross-validation and get the cross-validated accuracy scores
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Print the cross-validated accuracy scores, confusion matrix, and AUC score
print('Cross-validated accuracy scores:', cv_scores)
print('Confusion Matrix:\n', confusion)
print('AUC Score:', auc_score)

# Save the trained model to a file
joblib.dump(model, 'model/decision_tree_model.pkl')