import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import numpy as np
import pprint

#SVD analysis
def svd_analysis(file):
	df = pd.read_csv(file)
	odnames = list(df)
	u,s,vt = np.linalg.svd(df, full_matrices = True)
	pprint.pprint(s)


	newdf = pd.DataFrame(columns = range(12))
	

	newdf.to_csv('svd_analysis.csv')
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

svd_analysis('White Wine Qualities.csv')
svd_analysis('winequality-red.csv')

