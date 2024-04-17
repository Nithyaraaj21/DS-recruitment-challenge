import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from category_encoders import OrdinalEncoder
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
compressed_file_name = "./converted_dataset.csv.gz"
data = pd.read_csv(compressed_file_name, compression='gzip')

# Prepare data for predictive modeling
X = data[['Shop','BrandName', 'ModelGroup', 'ProductGroup']]
y = data['Returned']

# Preprocess categorical features using Ordinal Encoder
categorical_columns = ['Shop','BrandName', 'ModelGroup', 'ProductGroup']
ordinal_encoder = OrdinalEncoder()
X_encoded = ordinal_encoder.fit_transform(X)

# Apply SMOTE to balance the class distribution
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model (example: Decision Tree Classifier)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Serialize the trained model, ordinal encoder, and SMOTE
model_data = {
    'model': model,
    'ordinal_encoder': ordinal_encoder,
    'smote': smote
}

# Save the serialized objects to a pickle file
with open('model_data_dt.pkl', 'wb') as file:
    pickle.dump(model_data, file)

# Load the trained model, ordinal encoder, and SMOTE from the pickle file
with open('model_data_dt.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the model, ordinal encoder, and SMOTE from the loaded data
loaded_model = model_data['model']
loaded_ordinal_encoder = model_data['ordinal_encoder']
loaded_smote = model_data['smote']
