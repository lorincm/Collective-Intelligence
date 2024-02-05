import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the data
data = {
    'Agent': ['flocking(with-all)', 'limit', 'origin', 'stacking', 'stacking&limit'],
    'Fitness Score': [4.779661016949152, 3.3358208955223883, 3.207017543859649, 3.2606837606837606, 3.2467532467532467]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a figure and axis with a larger figure size
fig, ax = plt.subplots(figsize=(10, 5))

ax.set_ylim(2.6, 4.9)

# Create bar plot
sns.barplot(x='Agent', y='Fitness Score', data=df, ax=ax)

# Rotate x-axis labels for readability and adjust their alignment
plt.xticks(rotation=45, ha='right')

# Add title and show the plot
plt.title('Fitness Scores by Agent')
plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
plt.subplots_adjust(bottom=0.3)  # Add more padding at the bottom
plt.savefig('agentfitnessscores.png')
