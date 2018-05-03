import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# load data
df = pd.read_csv('winequality-red.csv')

#split into 'good' and 'bad' qualities
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])

#train dataset with response and feature variables
X = df.drop('quality', axis = 1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# run random forest alg
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))

print(confusion_matrix(y_test, pred_rfc))

print(pred_rfc)
print(len(y_test))
y_test.to_csv('forest_resuts.csv')
