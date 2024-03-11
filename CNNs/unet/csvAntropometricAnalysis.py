import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

dataPath ="D:/Resultados Antropometricos CSV/FatFraction3D/"
normalized = False
Matrix = []
df = pd.read_csv(dataPath + 'FatFraction3D.csv')
dfMale = df[df["Gender"] == "male"]
dfFemale = df[df["Gender"] == "female"]
dfC25K = df[(df["Group"] == "C25K") | (df["Group"] == "ACT10")]
dfCyclist = df[df["Group"] == "Cyclists"]
pltTitle = ["", "Hombres", "Mujeres", "Sedentarios", "Activos"]
dataFrames = [df, dfMale, dfFemale, dfC25K, dfCyclist]
lado = ["blue","red"]
nombreMusculos = ["Psoas","Iliaco","Cuadrado Lumbar","Erector Espinae y Multifido"]
num_labels = len(nombreMusculos)
tick_positions = [2 * i + 1.5 for i in range(num_labels)]
muscleNames = list(df.head(0))[-9:-1]
median_positions = [i + 1.4 for i in range(len(muscleNames))]
newDataFrames = []


for d, titles in zip(dataFrames, pltTitle):
    results = pd.DataFrame()
    if normalized:
        for muscle in muscleNames:
            results[muscle] = d[muscle].div(d['Peso (kg)'])
        AverageFFvalues = d["Promedio"].values/d['Peso (kg)'].values
    else:
        results = d[muscleNames]
        AverageFFvalues = d["Promedio"].values

    plt.figure(figsize=(15, 8))
    violin_parts = plt.violinplot(results, showmeans=True, showextrema=True)
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(lado[(i%2)])
    plt.xticks(tick_positions, nombreMusculos, fontsize=14)
    plt.title('Volumen (CM³) normalizado por peso' +' '+ titles, fontsize=16)


    medians = np.median(results, axis=0)
    for i, median in enumerate(medians):
        plt.text(median_positions[i], median, f'{median:.2f}', ha='center', va='bottom', fontsize=14)

    plt.savefig(dataPath + 'Volume_norm_Weight'+'_'+ titles+'.tif')
    plt.close()

    variableNames = list(d.head(0))[3:7]
    variables = d[variableNames].values
    variables = np.transpose(variables)
    for values, names in zip(variables, variableNames):

        slope, intercept, r_value, p_value, std_err = linregress(values, AverageFFvalues)
        r_squared = r_value ** 2

        # Write the equation and R-squared value
        equation = f'Y = {slope:.4f}X + {intercept:.2f}\nR² = {r_squared:.2f}'

        # Create the regression line using the slope and intercept
        regression_line = slope * values + intercept

        # Create a scatter plot of the data points
        plt.scatter(values, AverageFFvalues, color="blue")

        # Create the linear regression line plot
        plt.plot(values, regression_line, color="red")
        plt.text(np.max(values),np.min(AverageFFvalues), equation, ha='right', va='bottom')
        # Add labels and a legend
        plt.xlabel(names)
        plt.title('Volumen (CM³) promedio normalizado por Peso' + ' ' + titles)

        plt.savefig(dataPath + 'RL_volume_norm_Weight' + '_' + names + '_' + titles+'.tif')
        plt.close()
