import os
import requests
import pandas as pd

# Define the URL
url = "http://127.0.0.1:5000/analyze"

# Define the dataset folder
dataset_folder = "C:/Users/abcde/OneDrive/Documents/School/Year 4/Spring/Computer Vision/Hu-COSI159A/assignment_04/dataset/val"

# Initialize a list to store all predictions
all_predictions = []

# Iterate over each image in the dataset folder
for filename in os.listdir(dataset_folder):
    # Extract the file number from the filename
    file_number = os.path.splitext(filename)[0]

    # Construct the image path
    img_path = os.path.join(dataset_folder, filename)
    
    # Define the parameters
    params = {
        "img_path": img_path,
        "actions": ["age", "gender", "race"]
    }

    # Make the POST request
    response = requests.post(url, json=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the JSON response
        data = response.json()
        # Extract the results
        results = data.get("results", [])
        if results:
            # Extract the age, gender, and race from the first result
            prediction = results[0]
            age = prediction.get("age")
            gender = prediction.get("dominant_gender")
            race = prediction.get("dominant_race")
            # Add the prediction to the list
            all_predictions.append({"FileNumber": file_number, "Filename": filename, "Age": age, "Gender": gender, "Race": race})
        else:
            print(f"No results found for {filename}.")
    else:
        print(f"Failed to make the request for {filename}. Status code:", response.status_code)

# Convert the list of predictions to a DataFrame
predictions_df = pd.DataFrame(all_predictions)

# Save the predictions to a CSV file
predictions_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv.")
