# rainfall_model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt

# Generate dummy data
np.random.seed(42)
X = np.random.rand(100, 3)  # 3 features: temperature, humidity, wind speed
y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.1, 100)  # Rainfall

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'rainfall_model.joblib')

print("Model trained and saved.")

# Step 5: Visualize the data and model
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Plot each feature against the target
for i in range(3):
    axs[i//2, i%2].scatter(X[:, i], y, alpha=0.5)
    axs[i//2, i%2].set_xlabel(f'Feature {i+1}')
    axs[i//2, i%2].set_ylabel('Rainfall')
    axs[i//2, i%2].set_title(f'Feature {i+1} vs Rainfall')

# Plot predicted vs actual
y_pred = model.predict(X)
axs[1, 1].scatter(y, y_pred, alpha=0.5)
axs[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axs[1, 1].set_xlabel('Actual Rainfall')
axs[1, 1].set_ylabel('Predicted Rainfall')
axs[1, 1].set_title('Predicted vs Actual Rainfall')

plt.tight_layout()
plt.savefig('rainfall_model_visualization.png')
print("Visualization saved as 'rainfall_model_visualization.png'")