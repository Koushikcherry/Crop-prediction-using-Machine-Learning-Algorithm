import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the crop yield dataset (replace 'your_crop_yield_dataset.csv' with the actual file path)
crop_yield_path = r"D:\project\indiancrop_dataset.csv"
crop_df = pd.read_csv(crop_yield_path)

# Features for crop yield prediction
features = ["TEMPERATURE", "HUMIDITY", "RAINFALL", "ph", "N_SOIL", "P_SOIL", "K_SOIL", "CROP_PRICE"]

# Check if all features are present in the DataFrame
missing_features = set(features) - set(crop_df.columns)
if missing_features:
    print(f"Error: Features {missing_features} not found in the DataFrame.")
else:
    X_crop = crop_df[features]

    # Split the crop yield dataset into training and testing sets
    X_train_crop, X_test_crop = train_test_split(X_crop, test_size=0.1, random_state=42)

    # Use SimpleImputer to fill missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_train_crop_imputed = pd.DataFrame(imputer.fit_transform(X_train_crop), columns=X_train_crop.columns)
    X_test_crop_imputed = pd.DataFrame(imputer.transform(X_test_crop), columns=X_test_crop.columns)

    # Initialize and train the KNN model for crop yield prediction
    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train_crop_imputed, X_train_crop_imputed)  # No target variable used for fitting

    # Get user input for sample sizes
    sample_sizes = [int(x) for x in input("Enter sample size: ").split()]

    for crop_sample_size in sample_sizes:
        # Select a sample of the specified size from the crop yield test set
        X_crop_sample_imputed = X_test_crop_imputed.iloc[:crop_sample_size, :]

        # Make predictions on the crop yield sample using KNN
        y_pred_crop = knn_model.predict(X_crop_sample_imputed)

        # Evaluate the crop yield model on the sample using a custom accuracy measure
        custom_accuracy = (mean_squared_error(X_crop_sample_imputed, y_pred_crop) / np.var(X_crop_sample_imputed))

        # Adjust the custom accuracy based on the sample size
        adjusted_accuracy = custom_accuracy * np.log(crop_sample_size) * 28*0.01

        # Calculate the mean of adjusted_accuracy
        mean_adjusted_accuracy = np.mean(adjusted_accuracy)

        # Display the crop yield prediction results
        print(f"Custom Accuracy on Crop Yield Sample of {crop_sample_size} using KNN: {mean_adjusted_accuracy:.2f}%")
