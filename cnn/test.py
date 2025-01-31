import pandas as pd

folder_path = "D:\\Step-counter\\Output\\001"  # Change to an actual folder
left_file = f"{folder_path}\\001_left_acceleration_data.csv"
right_file = f"{folder_path}\\001_right_acceleration_data.csv"

left_data = pd.read_csv(left_file)
right_data = pd.read_csv(right_file)

print("\nLeft Data (First 5 rows):")
print(left_data.head())

print("\nRight Data (First 5 rows):")
print(right_data.head())

print("\nLeft Data Shape:", left_data.shape)
print("Righht Data Shape:", right_data.shape)