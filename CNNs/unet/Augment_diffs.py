import csv
import os
import matplotlib.pyplot as plt
import numpy as np


datapath = '..\\..\\Data\\ValidationCsv\\'

Augment = ['Standard', 'Linear', 'NonLinear', 'Augmented']
muscleNames = ['LM', 'LP','LQ','RM','RP','RQ']
Folders = [datapath + 'Standard', datapath + 'Linear', datapath + 'NonLinear', datapath + 'Augmented']

DiceMaxValues = [[] for n in range(len(Augment))]

for i, f in enumerate(Folders):                 #Augment Filenames
    muscles = os.listdir(f)         # Lists Muscles per Augment
    for m in muscles:
        filepath = os.path.join(f, m)
        with open(filepath) as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Skip the header row
            max_value = float('-inf')
            for row in csv_reader:
                value = float(row[1])
                max_value = max(max_value, value)
        DiceMaxValues[i].append(max_value)


DiceMaxValues = np.transpose(DiceMaxValues)
bar_width = 0.1

y_values1 = DiceMaxValues[0]
y_values2 = DiceMaxValues[1]
y_values3 = DiceMaxValues[2]
y_values4 = DiceMaxValues[3]
y_values5 = DiceMaxValues[4]
y_values6 = DiceMaxValues[5]

bar_width = 0.15

pos1 = np.arange(len(Augment))
pos2 = [x + bar_width for x in pos1]
pos3 = [x + bar_width for x in pos2]
pos4 = [x + bar_width for x in pos3]
pos5 = [x + bar_width for x in pos4]
pos6 = [x + bar_width for x in pos5]


fig, ax = plt.subplots()
ax.bar(pos1, y_values1, width=bar_width, label=muscleNames[0])
ax.bar(pos2, y_values2, width=bar_width, label=muscleNames[1])
ax.bar(pos3, y_values3, width=bar_width, label=muscleNames[2])
ax.bar(pos4, y_values4, width=bar_width, label=muscleNames[3])
ax.bar(pos5, y_values5, width=bar_width, label=muscleNames[4])
ax.bar(pos6, y_values6, width=bar_width, label=muscleNames[5])

ax.set_xticks(pos3)
ax.set_xticklabels(Augment)


ax.set_title('Validation Dice Scores')
ax.set_xlabel('Augment')
ax.set_ylabel('Score')
ax.set_ylim(0.9, 1)  # Set the lower and upper bounds of the y-axis
ax.legend()

plt.show()