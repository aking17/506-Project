import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import numpy as np
import pprint

#load RED WINE data
df = pd.read_csv('winequality-red.csv')

#clean the data (remove outliers)
df = df.loc[df['sulphates'] <= 1.7]
df = df.loc[df['chlorides'] <= .5]
df = df.loc[df['volatile acidity'] <= 1.5]


#load WHITE WINE data
df = pd.read_csv('winequality-red.csv')

#clean the data (remove outliers)
df = df.loc[df['sulphates'] <= 1.5]
df = df.loc[df['chlorides'] <= .5]
df = df.loc[df['volatile acidity'] <= 1.5]
print(df.shape)

cols = df.head(0)
newdf = pd.DataFrame(columns =cols )

# Filters out our data set based on the results of our features
def predictRedWines(dfname):

	newdf = pd.DataFrame(columns =cols )
	df1 = dfname
	newdf = df1.loc[df1['volatile acidity'] <= 0.39] #filter out everything over 25% -- ***.39***
	newdf = newdf.loc[newdf['sulphates'] >= 0.7]#filter out everything under 75%
	newdf = newdf.loc[newdf['chlorides'] <= 0.07]#filter out everything over 25%
	newdf = newdf.loc[newdf['sulfur dioxide'] >= 0.472]#filter out everything over 25%

	av = float(newdf['quality'].sum())/(len(newdf))
	return av, newdf


def predictWhiteWines(dfname):

	newdf = pd.DataFrame(columns =cols )
	df1 = dfname
	newdf = df1.loc[df1['volatile acidity'] <= 0.21] #filter out everything over 25% -- ***.39***
	#newdf = newdf.loc[newdf['sulphates'] >= 0.55]#filter out everything under 75%
	newdf = newdf.loc[newdf['chlorides'] <= 0.036]#filter out everything over 25%

	av = float(newdf['quality'].sum())/(len(newdf))
	return av, newdf
#print(newdf)

whitewine = pd.read_csv('White Wine Qualities.csv')
av, ndf = predictWhiteWines(whitewine)
av1, ndf1 = predictRedWines(df)
print('red wine predictions average: ' ,av1)
print('white wine predictions average: ' ,av)
#av, ndf = predictWines(whitewine)
def measureAcc(df):
	lst = df['quality'].tolist()
	#print(lst)
	great = 0
	good = 0
	av = 0
	trash = 0
	for i in lst:
		if i >= 7:
			great += 1.
		elif i == 6:
			good += 1.
		else:
			av += 1.
		

	div = len(lst)
	print(div)
	return(great/div, good/div, av/div)

print('red', measureAcc(ndf1),ndf1.shape)
print('white', measureAcc(ndf), ndf.shape)


whitewine = whitewine.loc[whitewine['quality']>=7]
print(whitewine.shape)
