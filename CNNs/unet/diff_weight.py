import csv
import os
import matplotlib.pyplot as plt
import numpy as np

datapath = '..\\..\\Data\\ModelDiff\\WeightDiff\\'

Weight = ['None', 'Standard', 'Relative']
weightAux = ['Sin Pesaje', 'Estándar', 'Relativo']
muscleNames = ['$ES+M_i$', '$ES+M_d$', '$CL_i$', '$CL_d$', '$P_i$', '$P_d$']
Folders = [datapath + Weight[0], datapath + Weight[1], datapath + Weight[2]]

DiceValues = [[] for n in range(len(Weight))]
BestVlossRow = []


for i, f in enumerate(Folders):        # Weight Filenames
    muscles = os.listdir(f)[:6]        # Lists Muscles per Augment
    sumValue = 0
    for m in muscles[:(len(muscleNames))]:
        filepath = os.path.join(f, m)
        with open(filepath) as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                Value = float(row[1])
                sumValue += Value
                DiceValues[i].append(Value)
    meanValue = sumValue / len(muscleNames)
    DiceValues[i].append(meanValue)

DiceValues = np.transpose(DiceValues)
bar_width = 0.1

y_values1 = DiceValues[0]
y_values2 = DiceValues[1]
y_values3 = DiceValues[2]
y_values4 = DiceValues[3]
y_values5 = DiceValues[4]
y_values6 = DiceValues[5]
y_values7 = DiceValues[6]

pos1 = np.arange(len(Weight))
pos2 = [x + bar_width for x in pos1]
pos3 = [x + bar_width for x in pos2]
pos4 = [x + bar_width for x in pos3]
pos5 = [x + bar_width for x in pos4]
pos6 = [x + bar_width for x in pos5]
pos7 = [x + bar_width for x in pos6]

fig, ax = plt.subplots()
ax.bar(pos1, y_values1, width=bar_width, label=muscleNames[0], color='darkgray')
ax.bar(pos2, y_values2, width=bar_width, label=muscleNames[1], color='lightcoral')
ax.bar(pos3, y_values3, width=bar_width, label=muscleNames[2], color='sandybrown')
ax.bar(pos4, y_values4, width=bar_width, label=muscleNames[3], color='khaki')
ax.bar(pos5, y_values5, width=bar_width, label=muscleNames[4], color='yellowgreen')
ax.bar(pos6, y_values6, width=bar_width, label=muscleNames[5], color='steelblue')
ax.bar(pos7, y_values7, width=bar_width, label='Promedio', color='black')

ax.set_xticks(pos4)
ax.set_xticklabels(weightAux)


ax.set_title('Puntaje Dice en Set de Validación')
ax.set_ylim(0.85, 1)  # Set the lower and upper bounds of the y-axis
ax.legend()

plt.savefig(datapath + 'WeightDiff.tiff')

