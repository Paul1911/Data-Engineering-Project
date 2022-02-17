# Todo: Add validity check visualisations

#Code from jupyter (no imports needed):
#Validity Checks

#Null values
#print(str(df.isnull().sum().max()) + " null values observed")

#Skewness
#print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
#print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

#Observations count, also per target class
#stdev, mean of transaction value