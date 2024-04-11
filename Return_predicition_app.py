import streamlit as st
import pandas as pd
import pickle
import gzip
import io

# Load the pickled model and preprocessing steps
with open('model_data.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the model and ordinal encoder from the loaded data
model = model_data['model']
ordinal_encoder = model_data['ordinal_encoder']

# Load the dataset
compressed_file_name = "youconverted_dataset.csv.gz"
with gzip.open(compressed_file_name, "rb") as f:
    # Read the compressed file as bytes
    compressed_bytes = f.read()
    # Convert the bytes to a file-like object
    compressed_file_object = io.BytesIO(compressed_bytes)
    # Read the CSV data from the file-like object
    data = pd.read_csv(compressed_file_object)

# Define a function to preprocess input data and make predictions
def predict_return(data):
    # Preprocess the input data using the ordinal encoder
    data_encoded = ordinal_encoder.transform(data)
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
    columns_to_select = ['BrandName', 'ModelGroup', 'ProductGroup']
    selected_values = []
    for column in columns_to_select:
        # Use unique values in the column as options for the dropdown menu
        options = data[column].unique()
        # Add a dropdown menu for the current column
        selected_value = st.selectbox(f'Select {column}', options=options)
        selected_values.append(selected_value)

    # Button to make predictions
    if st.button('Predict'):
        # Convert selected values to DataFrame
        input_data = {column: [selected_value] for column, selected_value in zip(columns_to_select, selected_values)}
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
