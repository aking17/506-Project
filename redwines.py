import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import numpy as np


#SVD analysis
def svd_analysis():
	df = pd.read_csv('winequality-red.csv')
	odnames = list(df)
	u,s,vt = np.linalg.svd(df)
	print(odnames)

	fig = plt.figure()
	plt.plot(range(1,1+len(s)),s)
	plt.xlabel(r'$k$',size=20)
	plt.ylabel(r'$\sigma_k$',size=20)
	_ = plt.title(r'Singular Values of $A$',size=20)
	plt.show()

	Anorm = np.linalg.norm(df)
	err = np.cumsum(s[::-1]**2)
	err = np.sqrt(err[::-1])
	plt.plot(range(1,13),err[:12]/Anorm)
	plt.xlim([0,12])
	plt.xlabel(r'$k$',size=16)
	plt.ylabel(r'relative F-norm error',size=16)
	_ = plt.title(r'Relative Error of rank-12',size=16)
	plt.show()

	print(err[:12]/Anorm)


	for i in range(12):
		ax = plt.subplot(4,3,i+1)
		plt.plot(df.iloc[:,i], markersize = 1)
		plt.title(odnames[i])
		plt.xlabel('Wine')
	plt.subplots_adjust(wspace=0.45,hspace=1)
	_ = plt.suptitle('Wine Qualities',size=20)
	plt.show()

#load data
df = pd.read_csv('winequality-red.csv')
#clean the data (remove outliers)
df = df.loc[df['sulphates'] <= 1.5]
df = df.loc[df['chlorides'] <= .5]
df = df.loc[df['volatile acidity'] <= 1.5]

#df_goodwine = pd.DataFrame()
#df_goodwine = df.loc[df['quality'] >= 7]
#create lists of our important variables
volatile = df['volatile acidity'].tolist()
chlorides = df['chlorides'].tolist()
density = df['density'].tolist()
sulphates = df['sulphates'].tolist()
quality = df['quality'].tolist()


#svd_analysis()

def ranges(lst):
	low = min(lst)
	high = max(lst)
	diff = high - low
	return diff
	#return 'low = ' + str(low) + ", high = " +str(high)+ ", difference = "+ str(diff)

def hist(lst):
	plt.hist(lst, bins='auto', rwidth =.7 , align='mid')
	plt.title ("Alcohol Histogram")
	plt.show()

def correlation_graph(feature, quality, name):
	m, b = np.polyfit(feature, quality, 1)

	line = [x*m +b for x in alc]

	plt.plot(feature, line, '-')
	plt.plot(feature,quality, 'ro', markersize=2)
	plt.xlabel(name)
	plt.ylabel("Quality")
	plt.title (name, " vs. Quality Graph")
	plt.show()

#create graph for volatile acidity feature
correlation_graph(volatile, quality, "Volatile Acidity"):

#create graph for volatile acidity feature
correlation_graph(chlorides, quality, "Chlorides"):

#create graph for density feature
correlation_graph(density, quality, "Density"):

#create graph for sulphates feature
correlation_graph(sulphates, quality, "Sulphates"):


# Filters out our data set based on the results of our features
def predictWines(dfname):
	#create data frame to store our recommended wine
	newdf = pd.DataFrame()

	df1 = dfname
	newdf = df1.loc[df1['volatile acidity'] <= 0.50] #filter out everything over 25% -- ***.39***
	newdf = newdf.loc[newdf['density'] <= 0.9956]#filter out everything over 25%
	newdf = newdf.loc[newdf['sulphates'] >= 0.73]#filter out everything under 75%
	newdf = newdf.loc[newdf['chlorides'] <= 0.07]#filter out everything over 25%
	av = float(newdf['quality'].sum())/(len(newdf))
	print( newdf)
	return av
whitewine = pd.read_csv('White Wine Qualities.csv')
print('red wine predictions average: ' ,predictWines(df))
print('white wine predictions average: ' ,predictWines(whitewine))
