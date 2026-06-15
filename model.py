"""
model.py — Exploratory Data Analysis only.
Run this script to inspect dataset statistics and distributions.
Training is handled exclusively by train_model.py.
"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

data = pd.read_csv('data/Crop_recommendation.csv')

print("Shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())
print("\nCrop distribution:\n", data['label'].value_counts())
print("\nFeature statistics:\n", data.describe())

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(data.drop('label', axis=1).corr(), annot=True, cmap='viridis', ax=ax)
ax.set_title('Feature Correlation')
plt.tight_layout()
plt.show()

# Feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
colors   = ['blue', 'green', 'darkblue', 'black', 'lightgreen', 'darkgreen', 'grey']

for ax, feat, color in zip(axes.flat, features, colors):
    sns.histplot(data[feat], color=color, kde=True, ax=ax)
    ax.set_xlabel(feat, fontsize=11)
    ax.grid(True)

axes.flat[-1].set_visible(False)
fig.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout()
plt.show()
