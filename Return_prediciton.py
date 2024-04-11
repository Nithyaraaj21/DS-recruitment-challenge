import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from category_encoders import OrdinalEncoder
import pickle

# Load the dataset
# Specify the compressed file name
compressed_file_name = "./converted_dataset.csv.gz"

# Read the compressed CSV file directly with pd.read_csv
data = pd.read_csv(compressed_file_name, compression='gzip')

# Prepare data for predictive modeling
X = data.drop(columns=["Returned", "SaleDocumentNumber", "CustomerID"])
y = data['Returned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical features using Ordinal Encoder
categorical_columns = ['BrandName', 'ModelGroup', 'ProductGroup', 'Shop']
ordinal_encoder = OrdinalEncoder()
X_train_encoded = ordinal_encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = ordinal_encoder.transform(X_test[categorical_columns])

# Train the model (example: Decision Tree Classifier)
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
