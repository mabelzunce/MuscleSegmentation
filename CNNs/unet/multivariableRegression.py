import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from sklearn.metrics import r2_score

dataPath ="D:/Resultados Antropometricos CSV/FatFraction3D/"
Matrix = []
df = pd.read_csv(dataPath + 'FatFraction3D.csv')
print(df.head(1))
dfMale = df[df["Genero"] == "male"]
dfFemale = df[df["Genero"] == "female"]
dfC25K = df[(df["Group"] == "C25K") | (df["Group"] == "ACT10")]
dfCyclist = df[df["Group"] == "Cyclists"]
pltTitle = ["", "Hombres", "Mujeres", "Sedentarios", "Activos"]
dataFrames = [df, dfMale, dfFemale, dfC25K, dfCyclist]

for d, titles in zip(dataFrames, pltTitle):
    muscleNames = list(d.head(1))[-9:-1]

    for muscle in muscleNames:
        Y = d[muscle].values
        variableNames = list(d.head(1))[2:6]
        matrix = np.transpose(d[variableNames].values)
        R = []
        for i, variables in enumerate(variableNames):
            # Fit a linear regression model
            X = np.column_stack(matrix[:i+1])
            model = LinearRegression()
            model.fit(X, Y)

            Y_pred_all = model.predict(X)
            r_squared = r2_score(Y, Y_pred_all)
            R.append(r_squared)
            with open(dataPath + "/resultados_" + titles + "_" + muscle + ".txt", "w") as file:
                for r in R:
                    writing = variables + ":" + str(r)
                    file.write(writing)


            #equation = f'Y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * X1 + {model.coef_[1]:.4f} * X2 + {model.coef_[2]:.4f} * X3'
            #r2_text = f'R-squared = {r_squared:.2f}'
        print("a")
        #X1_grid, X2_grid = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), np.linspace(X2.min(), X2.max(), 100))

