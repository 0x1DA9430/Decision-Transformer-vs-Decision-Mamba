import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Data from the table
data = {
    "Game": ["Breakout", "Qbert", "Hero", "KungFuMaster", "Pong", "Seaquest", "Alien", "BankHeist", "BattleZone", "RoadRunner"],
    "# Actions": [4, 6, 18, 14, 6, 18, 18, 18, 18, 18],
    "Avg. Trajectory Length": [1299.62, 1060.84, 1192.23, 2642.71, 2096.52, 1413.12, 932.20, 1185.34, 2068.26, 1123.01],
    "Avg. Total Reward per trajectory": [40.40, 131.87, 127.36, 117.24, 13.21, 61.45, 109.86, 58.85, 12.61, 129.36],
    "Avg. Steps to First non-zero Reward": [45.20, 56.75, 54.94, 109.53, 112.64, 87.23, 22.49, 20.04, 267.60, 81.03],
    "Image Entropy": [1.50, 1.89, 2.01, 2.66, 0.68, 2.24, 2.02, 1.88, 2.84, 1.77],
    "Feature Count": [23.33, 84.64, 38.84, 52.63, 9.16, 16.18, 22.88, 188.87, 13.88, 24.61]
}

df = pd.DataFrame(data)

# Set the figure size
plt.figure(figsize=(14, 8))

# Plotting bar plots for each metric
metrics = ["# Actions", "Avg. Trajectory Length", "Avg. Total Reward per trajectory", "Avg. Steps to First non-zero Reward", "Image Entropy", "Feature Count"]

fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 30))

for i, metric in enumerate(metrics):
    axs[i].bar(df["Game"], df[metric], color=plt.cm.Paired(np.arange(len(df))))
    axs[i].set_title(metric)
    axs[i].set_xlabel("Game")
    axs[i].set_ylabel(metric)
    axs[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('actions_bar_plot.png', dpi=300)
plt.close()


# Compute the correlation matrix
corr_matrix = df.drop(columns=["Game"]).corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 9))

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

# Set the title
plt.title('Heatmap of Correlations between Metrics')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()


# Create a grouped bar plot

# Setting up the data for grouped bar plot
metrics = ["Avg. Trajectory Length", "Avg. Total Reward per trajectory", "Avg. Steps to First non-zero Reward"]

# Normalize the data for better comparison
df_normalized = df.copy()
df_normalized[metrics] = df[metrics] / df[metrics].max()

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 8))

# Set bar width
bar_width = 0.2

# Set positions of the bars on the x-axis
r1 = np.arange(len(df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Make the plot
ax.bar(r1, df_normalized[metrics[0]], color='b', width=bar_width, edgecolor='grey', label=metrics[0])
ax.bar(r2, df_normalized[metrics[1]], color='r', width=bar_width, edgecolor='grey', label=metrics[1])
ax.bar(r3, df_normalized[metrics[2]], color='g', width=bar_width, edgecolor='grey', label=metrics[2])

# Add xticks on the middle of the group bars
ax.set_xlabel('Game', fontweight='bold')
ax.set_xticks([r + bar_width for r in range(len(df))])
ax.set_xticklabels(df['Game'], rotation=45)

# Adding labels and title
plt.ylabel('Normalized Value')
plt.title('Comparison of Metrics across Games')

# Create legend & Show graphic
plt.legend()
plt.tight_layout()
plt.savefig('grouped_bar_plot.png', dpi=300)
plt.close()
