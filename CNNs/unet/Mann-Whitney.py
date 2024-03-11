import pandas as pd
from scipy.stats import mannwhitneyu

side = True
gender = True
dataPath ="D:/Resultados Antropometricos CSV/Volumen/"
df = pd.read_csv(dataPath + 'Volumen.csv')
dfMale = df[df["Gender"] == "male"]
dfFemale = df[df["Gender"] == "female"]
dfCyclist = df[df["Group"] == "Cyclists"]
dfC25K = df[(df["Group"] == "C25K") | (df["Group"] == "ACT10")]



results = pd.DataFrame()
results["Mann-Whitney"] = ["U","P"]

if side:
    muscleNames = list(df.head(0))[-9:-1]
    CSV_name = "side"
    for k in range(0, len(muscleNames), 2):
        sample1 = df[muscleNames[k]].values
        sample2 = df[muscleNames[k + 1]].values
        estadistico_U, valor_p = mannwhitneyu(sample1, sample2)
        results[muscleNames[k]] = [estadistico_U, valor_p]
    results.to_csv(dataPath + 'mann_whitney_volume_' + CSV_name + '.csv', index=False)

else:
    muscleNames = list(df.head(0))[-9:]
    if gender:
        dataFrames = [dfMale.copy(), dfFemale.copy()]
        CSV_name = "gender"
    else:
        dataFrames = [dfCyclist.copy(), dfC25K.copy()]
        CSV_name = "group"
    results = pd.DataFrame()
    results["Mann-Whitney"] = ["U","P"]
    for muscle in muscleNames:
        sample1 = dataFrames[0][muscle].values/dataFrames[0]["Peso (kg)"].values
        sample2 = dataFrames[1][muscle].values/dataFrames[1]["Peso (kg)"].values
        estadistico_U, valor_p = mannwhitneyu(sample1, sample2)
        results[muscle] = [estadistico_U , valor_p]
    results.to_csv(dataPath + 'mann_whitney_norm_weight_' + CSV_name+'.csv', index=False)
