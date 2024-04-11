import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from category_encoders import OrdinalEncoder
import pickle
import gzip

# Load the dataset
# Specify the compressed file name
compressed_file_name = "youconverted_dataset.csv.gz"

# Save the DataFrame to a compressed CSV file
with gzip.open(compressed_file_name, "wt", compresslevel=9) as f:
    data = pd.read_csv(f)


# Prepare data for predictive modeling
X = data.drop(columns=["Returned", "SaleDocumentNumber", "CustomerID"])
y = data['Returned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical features using Ordinal Encoder
categorical_columns = ['BrandName', 'ModelGroup', 'ProductGroup']
ordinal_encoder = OrdinalEncoder()
X_train_encoded = ordinal_encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = ordinal_encoder.transform(X_test[categorical_columns])

# Train the model (example: Random Forest)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_encoded, y_train)

# Serialize the trained model and ordinal encoder
model_data = {
    'model': model,
    'ordinal_encoder': ordinal_encoder
}

# Save the serialized objects to a pickle file
with open('model_data.pkl', 'wb') as file:
    pickle.dump(model_data, file)