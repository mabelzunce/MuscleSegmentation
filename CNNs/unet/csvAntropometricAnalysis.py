import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

dataPath ="D:/Resultados Antropometricos CSV/CSA/"
Matrix = []
df = pd.read_csv(dataPath + 'CSA.csv')
print(df.head(1))
dfMale = df[df["Genero"] == "male"]
dfFemale = df[df["Genero"] == "female"]
dfC25K = df[(df["Group"] == "C25K") | (df["Group"] == "ACT10")]
dfCyclist = df[df["Group"] == "Cyclists"]
pltTitle = ["", "Hombres", "Mujeres", "Sedentarios", "Activos"]
dataFrames = [df, dfMale, dfFemale, dfC25K, dfCyclist]
lado = ["blue","red"]
nombreMusculos = ["Psoas","Cuadrado Lumbar","Erector Espinae & Multifido"]
num_labels = len(nombreMusculos)
tick_positions = [2 * i + 1.5 for i in range(num_labels)]
for d, titles in zip(dataFrames, pltTitle):
    muscleNames = list(d.head(1))[-7:-1]
    results = d[muscleNames]
    numRows = int(len(muscleNames)/2)
    plt.figure(figsize=(15, 8))
    violin_parts = plt.violinplot(results, showmeans=True, showextrema=True)
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(lado[(i%2)])
    plt.xticks(tick_positions, nombreMusculos)


    plt.title('Sección de Area Transversal (CM²) ' +' '+ titles)
    plt.savefig(dataPath + 'CSA'+'_'+ titles+'.tif')
    plt.close()

    AverageFFvalues = d["Promedio"].values
    variableNames = list(d.head(1))[2:6]
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
        plt.title('Sección de Area Transversal (CM²) Promedio' + ' ' + titles)

        plt.savefig(dataPath + 'RL_CSA_Promedio' + '_' + names + '_' + titles+'.tif')
        plt.close()
