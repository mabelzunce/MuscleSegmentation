private
double
GetFinalMetricFromElastixLogFile(string
filename)
{
    const
string
startOfMetricLine = "Final metric value  = ";
// const
string
registrationFinished = "Stopping condition: Maximum number of iterations has been reached.";
double
metricValue = Double.MinValue;
// Read
text
file:
// Need
to
look
for: Final
metric
value = -0.390318
string
line;
try
    {
        string
    bakFilename = outputPath + Path.GetFileNameWithoutExtension(logFilename) + ".bak";
    File.Copy(filename, bakFilename);
    StreamReader
    reader = new
    StreamReader(File.OpenRead(bakFilename)); // File.OpenRead, to
    open
    it as read
    only.
    // Read
    line:
    while ((line = reader.ReadLine()) != null)
    {
    if (line.StartsWith(startOfMetricLine))
    {
        metricValue = Convert.ToDouble(
        line.Substring(startOfMetricLine.Length, line.Length - startOfMetricLine.Length));
    }
    }
    reader.Close();
    // Remove
    the
    temporary
    file:
    File.Delete(bakFilename);
}
catch(Exception
exc)
{
lastError = exc.ToString();
return Double.NegativeInfinity;
}
return metricValue;
}