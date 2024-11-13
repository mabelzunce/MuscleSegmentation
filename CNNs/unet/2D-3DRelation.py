import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Genera plots con regresiones lineales entre los resultados 2d y 3d. Ya sea size o FF depende de que dataframe se toma en cuenta#
dataPath ="../../Data/ResultadosAntropometricosCSV/1CSA-Volume/"
Matrix = []
dfvolume = pd.read_csv(dataPath + 'Volumen.csv')
dfCSA = pd.read_csv(dataPath + "CSA.csv")
df3D = pd.read_csv(dataPath + "FatFraction3D.csv")
df2D = pd.read_csv(dataPath + "FatFractionL4.csv")
print(df3D.head(1))

muscleHeaders = list(df3D.head(1))[-7:]

nombreMusculos = ["Psoas izquierdo", "Psoas derecho","Cuadrado Lumbar izquierdo", "Cuadrado Lumbar derecho","Erector Espinae & Multifido izquierdo", "Erector Espinae & Multifido derecho", "Promedio"]

results2D = np.transpose(dfCSA[muscleHeaders].values)
results3D = np.transpose(dfvolume[muscleHeaders].values)
def format_2_decimal_places(x, pos):
    return f'{x:.2f}'
for f2, f3, names in zip(results2D,results3D, nombreMusculos):
    slope, intercept, r_value, p_value, std_err = linregress(f2, f3)
    r_squared = r_value ** 2

    # Write the equation and R-squared value
    equation = f'Y = {slope:.4f}X + {intercept:.2f}\nR² = {r_squared:.2f}'

    # Create the regression line using the slope and intercept
    regression_line = slope * f2 + intercept

    # Create a scatter plot of the data points
    plt.scatter(f2, f3, color="blue")

    # Create the linear regression line plot
    plt.plot(f2, regression_line, color="red")

    plt.text(np.max(f2), np.min(f3), equation, ha='right', va='bottom')

    # Add labels and a legend
    plt.xlabel("Resultados de sección transversal L4")
    plt.ylabel("Resultados volumetricos")
    plt.title('Tamaño' + ' ' + names)
    # Get the current axis
    ax = plt.gca()

    # Set the y-axis tick formatter to format ticks with 2 decimal places
    ax.yaxis.set_major_formatter(FuncFormatter(format_2_decimal_places))

    # Set the x-axis tick formatter to format ticks with 2 decimal places
    ax.xaxis.set_major_formatter(FuncFormatter(format_2_decimal_places))

    # Set the y-axis and x-axis tick intervals to 0.1
    #ax.yaxis.set_major_locator(MultipleLocator(0.05))
    #ax.xaxis.set_major_locator(MultipleLocator(0.05))


    plt.savefig(dataPath + 'RL_Tamaño' + '_' + names +'.tif')
    plt.close()

