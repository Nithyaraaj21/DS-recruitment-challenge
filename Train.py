import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from category_encoders import OrdinalEncoder
import pickle

# Load the dataset
compressed_file_name = "./converted_dataset.csv.gz"
data = pd.read_csv(compressed_file_name, compression='gzip')

# Prepare data for predictive modeling
X = data[['Shop','BrandName', 'ModelGroup', 'ProductGroup']]
y = data['Returned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical features using Ordinal Encoder
categorical_columns = ['BrandName', 'ModelGroup', 'ProductGroup']
ordinal_encoder = OrdinalEncoder()
X_train_encoded = ordinal_encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = ordinal_encoder.transform(X_test[categorical_columns])

# Train the model (example: Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_encoded, y_train)

# Serialize the trained model and ordinal encoder
model_data = {
    'model': model,
    'ordinal_encoder': ordinal_encoder
}

# Save the serialized objects to a pickle file
with open('model_data_rf.pkl', 'wb') as file:
    pickle.dump(model_data, file)

# Load the trained model and ordinal encoder from the pickle file
with open('model_data_rf.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the model and ordinal encoder from the loaded data
model = model_data['model']
ordinal_encoder = model_data['ordinal_encoder']

# Check column names of input data
print("Input Data Columns:")
print(X_test.columns)

# Check feature names used during training
print("Feature Names Used During Training:")
print(X_train_encoded.columns)

# Check categories mapped by the Ordinal Encoder
print("Categories Mapped by the Ordinal Encoder:")
for col, mapping in zip(categorical_columns, ordinal_encoder.category_mapping):
    print(f"{col}: {mapping}")
   

# Ensure number of columns is consistent between training and prediction datasets
if len(X_train_encoded.columns) == len(X_test_encoded.columns):
    print("Number of columns is consistent between training and prediction datasets.")
else:
    print("Number of columns is not consistent between training and prediction datasets.")

# Ensure column names are consistent between training and prediction datasets
if all(X_train_encoded.columns == X_test_encoded.columns):
    print("Column names are consistent between training and prediction datasets.")
else:
    print("Column names are not consistent between training and prediction datasets.")
