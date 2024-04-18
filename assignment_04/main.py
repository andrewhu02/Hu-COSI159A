import os
import pandas as pd
from matplotlib import pyplot as plt

# Define age and race ranges
AGE = {(0, 2): "0-2", (3, 9): "3-9", (10, 19): "10-19", (20, 29): "20-29",
       (30, 39): "30-39", (40, 49): "40-49", (50, 59): "50-59", (60, 69): "60-69", (70, float('inf')): "more than 70"}
race_names = ["White", "Black", "Asian", "Latino_Hispanic", "Middle Eastern", "Indian"]


# Load labels function
def load_labels(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

# Function to map age to age range
def map_age_to_range(age: int) -> str:
    for range_tuple, range_str in AGE.items():
        if isinstance(range_tuple, tuple):
            if range_tuple[0] <= age <= range_tuple[1]:
                return range_str
    return "Unknown"


def calculate_statistics(merged_df: pd.DataFrame):
    # Initialize statistics
    statistics = {
        "total": {"count": 0, "total_bins_difference": 0},
        "age_ranges": {range_str: {"count": 0, "total_bins_difference": 0} for range_str in AGE.values()},
        "gender": {"Male": {"count": 0, "total_bins_difference": 0}, "Female": {"count": 0, "total_bins_difference": 0}},
        "race": {"Matched": 0, "Mismatched": 0}
    }
    
    # Initialize lists for race statistics: total images, mismatches, and matches for each race
    race_total_images = [0] * len(race_names)
    race_mismatched_images = [0] * len(race_names)
    race_matched_images = [0] * len(race_names)
    
    # Process each row
    for _, row in merged_df.iterrows():
        # Determine age range and bins difference
        predicted_age_range = map_age_to_range(row['age_y'])
        bins_difference = 0
        
        # Age statistics
        statistics["total"]["count"] += 1
        
        # Calculate the actual age range index
        actual_age_range_index = None
        age_x_parts = row['age_x'].split('-')
        if 'more than' in age_x_parts[0].strip().lower():
            actual_age_range_index = (70, float('inf'))
        else:
            age_x_lower = int(age_x_parts[0])
            age_x_upper = int(age_x_parts[1])
            for age_range, range_str in AGE.items():
                if age_range[0] <= age_x_lower <= age_range[1] and age_range[0] <= age_x_upper <= age_range[1]:
                    actual_age_range_index = age_range
                    break

        predicted_age_range_index = [k for k, v in AGE.items() if v == predicted_age_range][0]

        bins_difference = abs(list(AGE.keys()).index(predicted_age_range_index) - list(AGE.keys()).index(actual_age_range_index))
        statistics["total"]["total_bins_difference"] += bins_difference

        # Update age range statistics
        statistics["age_ranges"][predicted_age_range]["count"] += 1
        statistics["age_ranges"][predicted_age_range]["total_bins_difference"] += bins_difference

        # Gender statistics
        gender = row['gender_x']
        statistics["gender"][gender]["count"] += 1
        statistics["gender"][gender]["total_bins_difference"] += bins_difference
        
        # Race statistics
        race_x = row['race_x'].strip() if isinstance(row['race_x'], str) else None
        race_y = row['race_y'].strip() if isinstance(row['race_y'], str) else None
        
        # Standardize race names
        if race_x.lower() in ['southeast asian', 'east asian']:
            race_x = 'Asian'
        if race_y.lower() in ['southeast asian', 'east asian']:
            race_y = 'Asian'
        
        # Update race statistics
        race_x_index = race_names.index(race_x)
        race_total_images[race_x_index] += 1
        
        if race_x == race_y:
            statistics["race"]["Matched"] += 1
            race_matched_images[race_x_index] += 1
        else:
            statistics["race"]["Mismatched"] += 1
            race_mismatched_images[race_x_index] += 1

    # Add the race mismatch data and race matched data to statistics
    statistics["race"]["race_mismatch"] = list(zip(race_names, race_total_images, race_mismatched_images, race_matched_images))
    
    return statistics

# Function to print statistics
def print_statistics(statistics):
    print("Statistics:")
    print("----------------")
    print(f"Total images analyzed: {statistics['total']['count']}")
    if statistics['total']['count'] > 0:
        print(f"Average bin difference: {statistics['total']['total_bins_difference'] / statistics['total']['count']:.2f}")
    else:
        print("Average bin difference: N/A (No images analyzed)")
    
    print("\nAge Range Statistics:")
    for age_range, stats in statistics['age_ranges'].items():
        if stats['count'] == 0:
            print(f"No images in Age Range {age_range}")
        else:
            print(f"Age Range {age_range}:")
            print(f"\tTotal images: {stats['count']}")
            print(f"\tAverage bin difference: {stats['total_bins_difference'] / stats['count']:.2f}")
    
    print("\nRace Statistics:")

    print("\nMismatch and Match Counts per Race:")
    for race, total_images, mismatched_images, matched_images in statistics["race"]["race_mismatch"]:
        print(f"Race: {race} - Mismatched: {mismatched_images} - Total images: {total_images} - Mismatch Rate: {(mismatched_images / total_images * 100):.2f}%")



if __name__ == "__main__":
    # Load ground truth labels
    val_df = load_labels("dataset/fairface_label_val.csv")

    # Load predictions from the CSV file
    predictions_df = pd.read_csv("predictions.csv")

    # Merge predictions with the ground truth based on 'file' column
    merged_df = val_df.merge(predictions_df, how='inner', on='file')
    merged_df = merged_df.drop(columns=['service_test'])

    # Calculate statistics
    statistics = calculate_statistics(merged_df)

    # Print statistics
    print_statistics(statistics)
