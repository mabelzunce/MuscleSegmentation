import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

gender = True #False = group
dataPath ="D:/PROYECTO FINAL/Resultados Antropometricos CSV/Volumen/"
Matrix = []
df = pd.read_csv(dataPath + 'Volumen.csv')
dfMale = df[df["Gender"] == "male"]
dfFemale = df[df["Gender"] == "female"]
dfCyclist = df[df["Group"] == "Cyclists"]
dfC25K = df[(df["Group"] == "C25K") | (df["Group"] == "ACT10")]


pltTitle = 'Fracción de tejido graso'
mannTitle = 'FatFraction'
normalized = False
if "Volumen" in dataPath:
    mannTitle = 'Volumen'
    pltTitle = 'Volumen (CM³) '
    normalized = True

if gender:
    subtitleViolin = " para distinto sexo"
    nameViolin = "gender"
    colores = ["green", "violet"]
    dataFrames = [dfMale.copy(), dfFemale.copy()]
else:
    subtitleViolin = " para distintos estilos de vida"
    nameViolin = "lifestyle"
    colores = ["grey", "orange"]
    dataFrames = [dfCyclist.copy(), dfC25K.copy()]


if normalized:
    subtitleNorm = " normalizado por altura"
    normName = "norm_Height_"
else:
    subtitleNorm = ""
    normName = ""


nombreMusculos = ["Psoas","Iliacus","Quadratus","Erector Spinae & Multifidus"]
num_labels = len(nombreMusculos)
tick_positions = [i + 1 for i in range(num_labels)]
muscleNames = list(df.head(0))[-9:]
median_positions = [i + 1.4 for i in range(len(muscleNames))]
newDataFrames=[]
tick_positions = [2 * i + 1.5 for i in range(num_labels)]
muscleNames = list(df.head(0))[-9:-1]
median_positions = [i + 1.4 for i in range(len(muscleNames))]


#gets the average of muscle pairs and creates a new dataframe#
for d in dataFrames:
    for k in range(0, len(muscleNames)-1, 2):
        d[nombreMusculos[int(k/2)]] = (d[muscleNames[k]] + d[muscleNames[k+1]])/2
    newDataFrames.append(d)


results = pd.DataFrame()
results[mannTitle] = ["U","P"]
p_values = []
values = []
for muscle in nombreMusculos:
    if normalized:
        sample1 = dataFrames[0][muscle].values/dataFrames[0]["Altura (cm)"].values
        sample2 = dataFrames[1][muscle].values/dataFrames[1]["Altura (cm)"].values
    else:
        sample1 = dataFrames[0][muscle].values
        sample2 = dataFrames[1][muscle].values
    values.append(sample1)
    values.append(sample2)
    estadistico_U, valor_p = mannwhitneyu(sample1, sample2)
    results[muscle] = [estadistico_U, valor_p]
    p_values.append(valor_p)
results.to_csv(dataPath + 'mann_whitney_' + normName + nameViolin +'.csv', index=False)



plt.figure(figsize=(15, 8))
violin_parts = plt.violinplot(values, showmeans=True, showextrema=True)
for i, pc in enumerate(violin_parts["bodies"]):
    pc.set_facecolor(colores[(i % 2)])
plt.xticks(tick_positions, nombreMusculos, fontsize=16)
plt.yticks(fontsize=16)

plt.title(pltTitle + subtitleNorm + subtitleViolin, fontsize=18)

medians = []
for k in values:
    medians.append(np.median(k))
for i, median in enumerate(medians):
    plt.text(median_positions[i], median, f'{median:.2f}', ha='center', va='bottom', fontsize=16)

plt.savefig(dataPath + mannTitle + '_' + normName + nameViolin + '_.tif')
plt.close()

