# %% [markdown]
# # Import Library

# %%
import numpy as np
import pandas as pd

# %%
# Membaca csv file
loan_data = pd.read_csv ('D:\A_ Project Data Science & Analyst\Credit Risk\loan_data_2007_2014.csv')

# %%
# Melihat keseluruhan datanya
loan_data

# %%
# Melihat info keseluruhan data
loan_data.info()

# Terdapat beberapa kolom/feature yang memiliki null values atau missing values

# %% [markdown]
# # Penentuan Target
# Dikarenakan project ini untuk mengetahui bad loan & good loan, maka perlu dibuat feature baru, yaitu target variable yang merepresentasikan bad loan (sebagai 1) dan good loan (sebagai 0).

# %%
# Melihat unique values pada feature loan_status
loan_data.loan_status.unique()

# %%
# Melihat kolom loan_status
loan_data[['loan_status']]

# %%
''' 
Membuat feature baru yaitu good_bad sebagai target variable, Jika loan_statusnya 'Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off' 
akan dianggap sebagai bad_loan atau 1 dan nilai selain itu akan dianggap good loan atau 0
'''
loan_data['good_bad'] = np.where(loan_data.loc[:, 'loan_status'].isin(['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off'])
                                 , 1, 0)

# %%
# Melihat distribusi 0 dan 1
loan_data.good_bad.value_counts()

# %%
loan_data[['loan_status', 'good_bad']]

# %% [markdown]
# # Missing Value
# Feature yang memiliki missing values lebih dari 50% akan di drop, karena jika ingin diisi dengan nilai lain seperti median atau mean, maka errornya akan sangat tinggi. Lebih baik di drop agar tidak membuat model semakin tidak akurat.

# %%
# # Melihat feature apa saja yang memiliki missing value lebih dari 50%
missing_values = pd.DataFrame (loan_data.isnull().sum()/loan_data.shape[0])
missing_values = missing_values[missing_values.iloc[:, 0] > 0.50]
missing_values.sort_values([0], ascending = False)

# %%
# # Drop feature tersebut
loan_data.dropna (thresh = loan_data.shape[0]*0.5, axis = 1, inplace = True)

# %%
# Pengececkan ulang apakah feature tersebut berhasil di drop
missing_values = pd.DataFrame (loan_data.isnull().sum()/loan_data.shape[0])
missing_values = missing_values[missing_values.iloc[:, 0] > 0.50]
missing_values.sort_values([0], ascending = False)

# %%
# Melihat shape data setelah di drop
loan_data.shape

# %% [markdown]
# # Data Splitting
# Data yang tersisa akan dibagi menjadi dua, yaitu untuk train dan test

# %%
# Import Library
from sklearn.model_selection import train_test_split

# %%
# Membagi data menjadi 80/20 dengan menyamakan distribusi dari bad loans di test set dengan train set.
X = loan_data.drop ('good_bad', axis= 1)
y = loan_data['good_bad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state=42)

# %%
y_train.value_counts(normalize= True)

# %%
y_test.value_counts(normalize= True)

# %% [markdown]
# # Data Cleaning

# %%
# Terdapat 54 kolom, selanjutnya kita akan membersihkan kolom yang memiliki data kotor
X_train.shape

# %%
# Terdapat 15 kolom yang memiliki tipe object dan boolean pada data
X_train.select_dtypes(include= ['object', 'bool'])

# %%
# Print untuk semua unique values kolom, sehingga dapat di cek satu-satu unique values apa saja yang kotor.
for col in X_train.select_dtypes(include= ['object', 'bool']).columns:
    print (col)
    print (X_train[col].unique())
    print ()

# %%
# Kolom/feature yang harus di cleaning
col_need_to_clean = ['term', 'emp_length', 'issue_d', 'earliest_cr_line', 'last_pymnt_d', 
                    'next_pymnt_d', 'last_credit_pull_d']

# %%
# Menghilangkan ' months' menjadi ''
X_train['term'].str.replace(' months', '')

# %%
# Convert data type menjadi numeric
X_train['term'] = pd.to_numeric (X_train['term'].str.replace(' months', ''))

# %%
X_train['term']

# %%
# Cek values apa saja yang harus di cleaning
X_train['emp_length'].unique()

# %%
X_train['emp_length'] = X_train['emp_length'].astype(str)

X_train['emp_length'] = X_train['emp_length'].str.replace('10+ years', '10')
X_train['emp_length'] = X_train['emp_length'].str.replace(' years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace('< 1 year', str(0))
X_train['emp_length'] = X_train['emp_length'].str.replace(' year', '')
X_train['emp_length'].replace('nan', np.nan, inplace=True)

X_train['emp_length'].fillna(value = 0, inplace=True)  
X_train['emp_length'] = pd.to_numeric(X_train['emp_length'])

# %%
X_train['emp_length'].unique()

# %%
# Cek feature date
col_date = ['issue_d', 'earliest_cr_line', 'last_pymnt_d',
                    'next_pymnt_d', 'last_credit_pull_d']

X_train[col_date]

# %%
for col in col_date:
    X_train[col] = pd.to_datetime(X_train[col], format='%b-%y')

# %%
X_train[col_need_to_clean].info()

# %%
X_test['term'].str.replace(' months', '')

# %%
X_test['term'] = pd.to_numeric (X_test['term'].str.replace(' months', ''))


# %%
X_test['emp_length'].unique()

# %%
# Lakukan hal yang sama untuk X_test
X_test['emp_length'] = X_test['emp_length'].astype(str)

X_test['emp_length'] = X_test['emp_length'].str.replace('10+ years', '10')
X_test['emp_length'] = X_test['emp_length'].str.replace(' years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace('< 1 year', str(0))
X_test['emp_length'] = X_test['emp_length'].str.replace(' year', '')
X_test['emp_length'].replace('nan', np.nan, inplace=True)

X_test['emp_length'].fillna(value = 0, inplace=True)  
X_test['emp_length'] = pd.to_numeric(X_test['emp_length'])

# %%
for col in col_date:
    X_test[col] = pd.to_datetime(X_test[col], format='%b-%y')

# %%
# Check apakah berhasil di cleaning
X_test[col_need_to_clean].info()

# %% [markdown]
# # Feature Enginering

# %%
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
# Kolom yang akan di feature engineering
col_need_to_clean

# %%
X_train = X_train[col_need_to_clean]
X_test = X_test[col_need_to_clean]

# %%
# tidak dibutuhkan untuk feature engineering
del X_train['next_pymnt_d']
del X_test['next_pymnt_d']

# %%
X_train.shape, X_test.shape

# %%
from datetime import date

date.today().strftime('%Y-%m-%d')

# %%
# feature engineering untuk date columns
def date_columns(df, column):
    today_date = pd.to_datetime(date.today().strftime('%Y-%m-%d'))
    df[column] = pd.to_datetime(df[column], format="%b-%y")
    df['mths_since_' + column] = (today_date.year - df[column].dt.year) * 12 + today_date.month - df[column].dt.month
    df.drop(columns=[column], inplace=True)

# %%
# apply to X_train
date_columns(X_train, 'earliest_cr_line')
date_columns(X_train, 'issue_d')
date_columns(X_train, 'last_pymnt_d')
date_columns(X_train, 'last_credit_pull_d')

# %%
# apply to X_test
date_columns(X_test, 'earliest_cr_line')
date_columns(X_test, 'issue_d')
date_columns(X_test, 'last_pymnt_d')
date_columns(X_test, 'last_credit_pull_d')

# %%
X_train.isnull().sum()

# %%
X_test.isnull().sum()

# %%
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)

# %%
X_train.isnull().sum()

# %%
X_test.isnull().sum()

# %% [markdown]
# # Modelling

# %%
from sklearn.linear_model import LogisticRegression

# %%
model = LogisticRegression()

# %%
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
result = pd.DataFrame(list(zip(y_pred,y_test)), columns = ['y_pred', 'y_test'])
result.head()

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# %%
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# %%
cm = confusion_matrix(y_test, y_pred)


sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')

plt.show()

# %%
# Hasil di atas adalah hasil yang misleading karena data yang digunakan adalah imbalance data, oleh sebab itu perlu pengukuran ROC dan AUC 
y_train.value_counts(normalize=True)

# %% [markdown]
# # ROC & AUC

# %%
model.predict(X_test)

# %%
# memprediksi probability dan mengambil probability kelas positive
y_pred = model.predict_proba(X_test)[:, 1]
y_pred

# %%
(y_pred > 0.5).astype(int)

# %%
# distribusi predicted probability
plt.hist(y_pred);

# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# %%
# youden j-statistic
j = tpr - fpr

ix = np.argmax(j)

best_thresh = thresholds[ix]
best_thresh

# %%
y_pred = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred > 0.110).astype(int)

# %%
cm = confusion_matrix(y_test, y_pred)


sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')

plt.show()

# %%
model.coef_

# %%
model.intercept_

# %%
df_coeff = pd.DataFrame(model.coef_, columns=X_train.columns)
df_coeff

# %%
X_train.head()

# %% [markdown]
# # Modelling II

# %%
from imblearn.over_sampling import SMOTE

# Menggunakan SMOTE untuk oversampling kelas minoritas pada data training
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Inisialisasi model
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# Melatih model Random Forest
rf_model.fit(X_train_resampled, y_train_resampled)
rf_predictions = rf_model.predict(X_test)

# Melatih model Gradient Boosting
gb_model.fit(X_train_resampled, y_train_resampled)
gb_predictions = gb_model.predict(X_test)

# Evaluasi Random Forest
print("Random Forest Classifier")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# Evaluasi Gradient Boosting
print("Gradient Boosting Classifier")
print(confusion_matrix(y_test, gb_predictions))
print(classification_report(y_test, gb_predictions))




