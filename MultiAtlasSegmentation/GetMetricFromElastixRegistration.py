import sys
import os

def GetFinalMetricFromElastixLogFile(fullFilename):
    startOfMetricLine = "Final metric value  = "
    registrationFinished = "Stopping condition: Maximum number of iterations has been reached."
    metricValue = sys.float_info.min
    # Read file line by line:

    #try
    if os.path.exists(fullFilename):
        f = open(fullFilename, "r")
        if f.mode == 'r':
            lines = f.readlines()
            f.close()
            for line in lines:
                if line.startswith(startOfMetricLine):
                    metricValue = float(line[startOfMetricLine.__len__() : ])
            return metricValue
    else:
        return float('inf')
