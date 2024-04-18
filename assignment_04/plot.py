import matplotlib.pyplot as plt

# Data
races = ['White', 'Asian', 'Black', 'Indian', 'Mid. East', 'Latino']
mismatch_rates = [17.8, 21.7, 41.1, 57.7, 64.5, 65.1]  

# Plot
plt.figure(figsize=(10, 6))  
plt.bar(races, mismatch_rates, color='tab:blue', alpha=0.7)
plt.xlabel('Race')
plt.ylabel('Mismatch Rate (%)')
plt.title('DeepFace Mismatch Rate Per Race (n = 3558)')
plt.ylim(0, 100)
plt.show()

# Age range statistics
age_ranges = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", ">70"]
total_images = [0, 0, 1, 1921, 1529, 105, 2, 0, 0]
average_bin_difference = [0, 0, 1.00, 1.02, 1.16, 1.29, 1.00, 0, 0]

# Create histogram
plt.figure(figsize=(10, 6))
plt.bar(age_ranges, average_bin_difference, color='orange', label='Average Bin Difference', alpha=0.5)

# Add labels and title
plt.xlabel('Age Range')
plt.ylabel('Average Bin Difference')
plt.title('Average Bin Distance Per Age Range (n = 3558)')
plt.legend()

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()