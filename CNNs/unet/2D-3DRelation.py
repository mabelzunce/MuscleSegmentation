import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Genera plots con regresiones lineales entre los resultados 2d y 3d. Ya sea size o FF depende de que dataframe se toma en cuenta#
dataPath ="D:/Resultados Antropometricos CSV/1.CSA-Volume/"
fatfraction = True  #False = volume


Matrix = []
dfVolume = pd.read_csv(dataPath + 'Volumen.csv')
dfCSA = pd.read_csv(dataPath + "CSA.csv")
df3D = pd.read_csv(dataPath + "FatFraction3D.csv")
df2D = pd.read_csv(dataPath + "FatFractionL4.csv")

nombreMusculos = ["P Izq", "P Der","CL Izq", "CL Der","ES+M Izq", "ES+M Der", "Promedio"]
musculos = ["Psoas", "Cuadrado Lumbar", "Erector Spinae y Multifido"]
#gets the average of muscle pairs and creates a new dataframe#
df3DAux = pd.DataFrame()
df2DAux = pd.DataFrame()


# Iterate over rows in the existing DataFrames
#gets the average of muscle pairs and creates a new dataframe#



if fatfraction:
    for k in range(0, len(nombreMusculos) - 1, 2):
        df3DAux[musculos[int(k / 2)]] = (df3D[nombreMusculos[k]] + df3D[nombreMusculos[k + 1]]) / 2
        df2DAux[musculos[int(k / 2)]] = (df2D[nombreMusculos[k]] + df2D[nombreMusculos[k + 1]]) / 2
    muscleHeaders = list(df2DAux.head(0))[-3:]
    results2D = np.transpose(df2DAux[muscleHeaders].values)
    results3D = np.transpose(df3DAux[muscleHeaders].values)
    figname = 'FatFraction'
    xlabel = 'Resultados bidimensionales'
    ylabel = 'Resultados tridimensionales'
    title = 'Fracción de tejido graso'

    def format_2_decimal_places(x, pos):
        return f'{x:.2f}'

else:
    for k in range(0, len(nombreMusculos) - 1, 2):
        df3DAux[musculos[int(k / 2)]] = (dfVolume[nombreMusculos[k]] + dfVolume[nombreMusculos[k + 1]]) / 2
        df2DAux[musculos[int(k / 2)]] = (dfCSA[nombreMusculos[k]] + dfCSA[nombreMusculos[k + 1]]) / 2
    muscleHeaders = list(df2DAux.head(0))[-3:]
    results2D = np.transpose(df2DAux[muscleHeaders].values)
    results3D = np.transpose(df3DAux[muscleHeaders].values)
    figname = 'Size'
    xlabel = 'Resultados de sección transversal'
    ylabel = 'Resultados volumetricos'
    title = 'Tamaño'


for r2D, r3D, names in zip(results2D, results3D, musculos):

    slope, intercept, r_value, p_value, std_err = linregress(r2D, r3D)
    r_squared = r_value ** 2

    # Write the equation and R-squared value
    equation = f'Y = {slope:.4f}X + {intercept:.2f}\nR² = {r_squared:.2f}'

    # Create the regression line using the slope and intercept
    regression_line = slope * r2D + intercept

    # Create a scatter plot of the data points
    plt.scatter(r2D, r3D, color="blue")

    # Create the linear regression line plot
    plt.plot(r2D, regression_line, color="red")

    plt.text(np.max(r2D), np.min(r3D), equation, ha='right', va='bottom',fontsize=12)

    # Add labels and a legend
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title + ' ' + names,fontsize=16)
    if fatfraction:
        # Get the current axis
        ax = plt.gca()

        # Set the y-axis tick formatter to format ticks with 2 decimal places
        ax.yaxis.set_major_formatter(FuncFormatter(format_2_decimal_places))

        # Set the x-axis tick formatter to format ticks with 2 decimal places
        ax.xaxis.set_major_formatter(FuncFormatter(format_2_decimal_places))

        # Set the y-axis and x-axis tick intervals to 0.1
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_locator(MultipleLocator(0.05))

    plt.savefig(dataPath + 'RL_'+ figname + '_' + names +'.tif')
    plt.close()

