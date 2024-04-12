import streamlit as st
import pandas as pd
import pickle
import gzip
import io
import numpy as np  # Import NumPy for array manipulation

# Load the pickled model and preprocessing steps
with open('model_data.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the model and ordinal encoder from the loaded data
model = model_data['model']
ordinal_encoder = model_data['ordinal_encoder']

# Load the dataset
compressed_file_name = "converted_dataset.csv.gz"
with gzip.open(compressed_file_name, "rb") as f:
    # Read the compressed file as bytes
    compressed_bytes = f.read()
    # Convert the bytes to a file-like object
    compressed_file_object = io.BytesIO(compressed_bytes)
    # Read the CSV data from the file-like object
    data = pd.read_csv(compressed_file_object)

# Define categorical columns
categorical_columns = ['Shop', 'BrandName', 'ModelGroup', 'ProductGroup']

# Define a function to preprocess input data and make predictions
def predict_return(data):
    # Reorder columns to match the order used during training
    data_reordered = data[['Shop', 'BrandName', 'ModelGroup', 'ProductGroup']]
    # Preprocess the input data using the ordinal encoder
    data_encoded = ordinal_encoder.transform(data_reordered)
    # Reshape the data to match the expected input shape of the model
    data_encoded = np.array(data_encoded)  # Convert to NumPy array
    data_encoded = data_encoded.reshape(1, -1)  # Reshape to a single row
    # Make predictions using the trained model
    predictions = model.predict(data_encoded)
    return predictions

# Streamlit app
def main():
    # Title of the app
    st.title('Return Prediction App')

    # Input form for user to enter data
    st.header('Enter Customer Data')

    # Create dropdown menus for the specified columns
    selected_values = []
    for column in categorical_columns:
        # Use unique values in the column as options for the dropdown menu
        options = data[column].unique()
        # Add a dropdown menu for the current column
        selected_value = st.selectbox(f'Select {column}', options=options)
        selected_values.append(selected_value)

    # Button to make predictions
    if st.button('Predict'):
        # Convert selected values to DataFrame
        input_data = {column: [selected_value] for column, selected_value in zip(categorical_columns, selected_values)}
        input_df = pd.DataFrame(input_data)

        # Preprocess input data and make predictions
        predictions = predict_return(input_df)

        # Display prediction result
        if predictions[0] == 0:
            st.write('Product is not returned')
        else:
            st.write('Product is returned')

if __name__ == '__main__':
    main()
