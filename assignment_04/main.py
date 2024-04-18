import os
import pandas as pd
from matplotlib import pyplot as plt

# Define age and race ranges
AGE = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
       "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8}

RACE = {"White": 0, "Indian": [1, 3], "East Asian": 2, "Southeast Asian": 3,
        "Latino_Hispanic": 4, "Middle Eastern": 5, "Black": 6}

# Load labels function
def load_labels(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

# Function to map age to age range
def map_age_to_range(age: int) -> int:
    for range_str, range_value in AGE.items():
        if isinstance(range_value, tuple):
            if range_value[0] <= age <= range_value[1]:
                return range_value
        else:
            if age == range_value:
                return range_value
    return -1  # return -1 or any other value to indicate "Unknown"

# Function to map race to race category
def map_race_to_category(race: str) -> str:
    race_lower = race.lower()
    for race_category, race_values in RACE.items():
        if isinstance(race_values, list):
            if race_lower in [rv.lower() for rv in race_values]:
                return race_category
        else:
            if race_lower == race_category.lower():
                return race_category
    return "Unknown"


def calculate_statistics(predictions_df: pd.DataFrame, merged_df: pd.DataFrame):
    statistics = {
        "total": {"count": 0, "total_bins_difference": 0},
        "age_ranges": {i: {"count": 0, "total_bins_difference": 0} for i in range(9)},
        "gender": {"Male": {"count": 0, "total_bins_difference": 0}, "Female": {"count": 0, "total_bins_difference": 0}},
        "race": {"Matched": 0, "Mismatched": 0, "Asian_matched": 0}
    }

    if 'race' in merged_df.columns:  # Check if 'race' column is present
        for index, row in predictions_df.iterrows():
            age = row['age']
            age_range = map_age_to_range(age)
            bins_difference = 0  # Initialize bins_difference to 0

            # Age statistics
            statistics["total"]["count"] += 1
            predicted_age = row['age']
            predicted_age_range = map_age_to_range(predicted_age)
            if predicted_age_range != -1:
                predicted_age = int(predicted_age)
                predicted_age_range = map_age_to_range(predicted_age)
                bins_difference = abs(age_range - predicted_age_range)
                statistics["total"]["total_bins_difference"] += bins_difference
                statistics["age_ranges"][age_range]["count"] += 1
                statistics["age_ranges"][age_range]["total_bins_difference"] += bins_difference

            # Gender statistics
            gender = row['gender']
            gender = gender.replace("Man", "Male")  # Change "Man" to "Male"
            statistics["gender"][gender]["count"] += 1
            statistics["gender"][gender]["total_bins_difference"] += bins_difference

            # Race statistics
            race_x = row['race_x'].strip()  # Remove leading and trailing whitespaces
            race_y = row['race_y'].strip()  # Remove leading and trailing whitespaces
            if race_x == race_y:
                statistics["race"]["Matched"] += 1
            else:
                statistics["race"]["Mismatched"] += 1
            if race_y.lower() in ['asian', 'east asian', 'southeast asian']:
                statistics["race"]["Asian_matched"] += 1
    else:
        print("Warning: 'race' column not found in merged DataFrame.")

    return statistics


# Function to print statistics
def print_statistics(statistics):
    print("Statistics:")
    print("----------------")
    print(f"Total images analyzed: {statistics['total']['count']}")
    if statistics['total']['count'] > 0:
        print(f"Average absolute age difference: {statistics['total']['total_bins_difference'] / statistics['total']['count']:.2f}")
    else:
        print("Average absolute age difference: N/A (No images analyzed)")
    print("\nAge Range Statistics:")
    for age_range, stats in statistics['age_ranges'].items():
        if stats['count'] == 0:
            print(f"No images in Age Range {age_range * 10}-{(age_range + 1) * 10 - 1}")
        else:
            print(f"Age Range {age_range * 10}-{(age_range + 1) * 10 - 1}:")
            print(f"\tTotal images: {stats['count']}")
            print(f"\tAverage absolute age difference: {stats['total_bins_difference'] / stats['count']:.2f}")

    print("\nGender Statistics:")
    for gender, stats in statistics['gender'].items():
        print(f"Gender: {gender}")
        print(f"\tTotal images: {stats['count']}")
        print(f"\tAverage absolute age difference: {stats['total_bins_difference'] / stats['count']:.2f}")

    print("\nRace Statistics:")
    print(f"Matched: {statistics['race']['Matched']}")
    print(f"Mismatched: {statistics['race']['Mismatched']}")
    print(f"Asian Matched: {statistics['race']['Asian_matched']}")


if __name__ == "__main__":
    # Load ground truth labels
    val_df = load_labels("dataset/fairface_label_val.csv")
    print(val_df.head())

    # Load predictions from the CSV file
    predictions_df = pd.read_csv("predictions.csv")
    print(predictions_df.head())

    # Merge predictions with the ground truth based on 'file' column
    merged_df = val_df.merge(predictions_df, how='left', on='file')
    print(merged_df.head())

    # Calculate statistics
    statistics = calculate_statistics(predictions_df, merged_df)

    # Print statistics
    print_statistics(statistics)
