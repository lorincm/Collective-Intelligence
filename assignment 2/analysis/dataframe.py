import matplotlib.pyplot as plt

# Results data
# data = [("Stable/Cyclic", [1.05, 0.2, 0.05, 0.2]),
#         ("All predator dead", [1.1, 0.3, 0.1, 0.3]),
#         ("All predator dead", [1.15, 0.4, 0.15, 0.4]),
#         ("Stable/Cyclic", [1.2, 0.5, 0.2, 0.5]),
#         ("Stable/Cyclic", [1.05, 0.5, 0.05, 0.5]),
#         ("All predator dead", [1.1, 0.4, 0.1, 0.4]),
#         ("All predator dead", [1.15, 0.3, 0.15, 0.3]),
#         ("Stable/Cyclic", [1.2, 0.2, 0.2, 0.2])]

data=  [('Ecosystem deceased', [0.001, 1.0]),
        ('Ecosystem deceased', [0.005, 1.0]),
        ('Ecosystem deceased', [0.005, 0.5]),
        ('Ecosystem deceased', [0.01, 0.5]),
        ('Ecosystem deceased', [0.01, 0.25])]


# Create figure and axes
fig, ax = plt.subplots(1,1)

# Hide the axes
ax.axis('tight')
ax.axis('off')

# Create table and add it to the axes
table_data = [("Energy Params", "Outcome")] + [(d[1], ', '.join(map(str, d[0]))) for d in data]
table = ax.table(cellText=table_data, colLabels=None, cellLoc = 'center', loc='center')

# Adjust the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save the figure
plt.savefig("Assignment2/analysis/results_table.png")
