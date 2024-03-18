import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000

height_mean = 170
height_std = 10
height = np.random.normal(height_mean, height_std, num_samples)

weekly_sugar_intake_mean = 150
weekly_sugar_intake_std = 50
weekly_sugar_intake = np.random.normal(weekly_sugar_intake_mean, weekly_sugar_intake_std, num_samples)

weekly_activity_mean = 3
weekly_activity_std = 1
weekly_activity = np.random.normal(weekly_activity_mean, weekly_activity_std, num_samples)

weight_mean = 70
weight_std = 10
weight = np.random.normal(weight_mean, weight_std, num_samples)

age_mean = 40
age_std = 10
age = np.random.normal(age_mean, age_std, num_samples)

# Generate labels (0 for non-diabetic, 1 for diabetic)
labels = np.random.randint(2, size=num_samples)

# Create DataFrame
data = pd.DataFrame({
    'Height': height,
    'Weekly Sugar Intake': weekly_sugar_intake,
    'Weekly Activity': weekly_activity,
    'Weight': weight,
    'Age': age,
    'Diabetic': labels
})

# Save DataFrame to CSV
data.to_csv('diabetes_dataset.csv', index=False)