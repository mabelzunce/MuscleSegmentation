import csv
import numpy as np
import matplotlib.pyplot as plt
dataPath ="../../../MuscleSegmentation/"
Matrix = []
with open(dataPath + 'Fatfraction3D.csv', mode='r', newline="") as csvfile:
    # Create a CSV reader
    csvreader = csv.reader(csvfile)
    # Iterate through the rows in the CSV file
    for row in csvreader:
        Matrix.append(row)
        print(row)
    MuscleNames = Matrix[0][1:]
    Matrix = np.transpose(np.array(Matrix)[1:])[1:]
    numRows = len(MuscleNames)

    plt.figure(figsize=(15,10))
    plt.violinplot(np.transpose(Matrix.astype(np.float64)), showmeans=True, showextrema=True)

    # Add labels to the x-axis ticks
    plt.xticks(range(1, numRows + 1), MuscleNames)

    # Add labels for the y-axis and title
    #plt.ylabel('CM³')
    plt.title('Fracción de tejido Graso')
    plt.savefig(dataPath + 'FF3D.tif')
