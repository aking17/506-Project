import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import numpy as np
import pprint

#load red wine
df = pd.read_csv('winequality-red.csv')

#clean the data (remove outliers)
df = df.loc[df['sulphates'] <= 1.7]
df = df.loc[df['chlorides'] <= .5]
df = df.loc[df['volatile acidity'] <= 1.5]

#create list of data for correlation graphs
volatile = df['volatile acidity'].tolist()
chlorides = df['chlorides'].tolist()
density = df['density'].tolist()
sulphates = df['sulphates'].tolist()
quality = df['quality'].tolist()


whitewine = pd.read_csv('White Wine Qualities.csv')


#create list of data for correlation graphs for whites

volatile_white = whitewine['volatile acidity'].tolist()
sulphates_white = whitewine['sulphates'].tolist()
quality_white = whitewine['quality'].tolist()
sulphates_white = whitewine['sulphates'].tolist()
chlorides_white = whitewine['chlorides'].tolist()


def correlation_graph(feature, quality, name):
	m, b = np.polyfit(feature, quality, 1)

	line = [x*m +b for x in feature]

	plt.plot(feature, line, '-')
	plt.plot(feature,quality, 'ro', markersize=2)
	plt.xlabel(name)
	plt.ylabel("Quality")
	plt.title (name+ " vs. Quality Graph")
	plt.show()

#create graph for volatile acidity feature
correlation_graph(volatile, quality, "Volatile Acidity")

#create graph for volatile acidity feature
correlation_graph(chlorides, quality, "Chlorides")

#create graph for density feature
correlation_graph(density, quality, "Density")

#create graph for sulphates feature
correlation_graph(sulphates, quality, "Sulphates")

#white wine correlation graphs
correlation_graph(volatile_white,quality_white, 'volatile Acidity')
correlation_graph(sulphates_white,quality_white, 'Sulphates')
correlation_graph(chlorides_white,quality_white, 'Chlorides')