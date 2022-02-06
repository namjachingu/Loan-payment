import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import load_model

import random
random.seed(101)

df = pd.read_csv("lending_club_loan_two.csv")

### Examining data and attributes ###
df.info()

sns.countplot(x="loan_status", data=df)

plt.figure(figsize=(12,2))
sns.displot(x="loan_amnt", data=df, kde=False, bins=40)

# Heatmap of correlation between continuous feature variables
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap="plasma")
plt.ylim(10, 0)

# Correlation between installment and loan amount as a scatterplot.
sns.scatterplot(x="installment", y="loan_amnt", data=df)

# Boxplot illustrating the relationship between loan status and loan amount.
sns.boxplot(x="loan_status", y="loan_amnt", data=df)

#Calculating the summary statistics for the loan amount, grouped by loan status.
df.groupby('loan_status')['loan_amnt'].describe()

#Countplot per grade
sns.countplot(sorted(df["grade"]), data=df, hue="loan_status")


#Display countplot of subgrades, by all loans and then separated based on loan status.
plt.figure(figsize=(20,12))
sns.countplot(x=sorted(df['sub_grade']), data=df, palette="summer")

plt.figure(figsize=(20,12))
sns.countplot(x=sorted(df['sub_grade']), data=df, hue="loan_status", palette="summer")

df_2 = df[(df['grade'] == "F") | (df['grade'] == "G")]

#F and G subgrades do not pay back that often, hence exploring those features:
plt.figure(figsize=(20,12))
sub_order = sorted(df_2["sub_grade"].unique())
sns.countplot(x="sub_grade", data=df_2, order=sub_order, hue="loan_status", palette="summer")

#Adding new column, loan repaid, where if loan status is fully repaid it will return 1 and 0 otherwise. 
def repaid(status):
    if status.lower() == "fully paid":
        return 1
    else:
        return 0
 
df["loan_repaid"] = df["loan_status"].apply(repaid)
df[["loan_status", "loan_repaid"]].head()


### Data preprocessing ###
missing_val = df.isnull().sum()
missing_val_percentage = (100*df.isnull().sum())/len(df)

df["emp_title"].nunique()
df[ "emp_length"].nunique()

df.drop("emp_title", axis=1, inplace=True)

sorted(df['emp_length'].dropna().unique())
sorted_emp = ['< 1 year', '1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']
plt.figure(figsize=(20, 12))
sns.countplot(x="emp_length", data=df, order=sorted_emp)

plt.figure(figsize=(20,12))
sns.countplot(x="emp_length", data=df, order=sorted_emp, hue="loan_status", palette="winter")

# Checking in percentage how many people did not pay back their loan with regards to employment years.
charged_off = df[df['loan_status'] == "Charged Off"].groupby("emp_length").count()['loan_status']
fully_paid = df[df['loan_status'] == "Fully Paid"].groupby("emp_length").count()['loan_status']
employment_length = charged_off/fully_paid

df.drop(["emp_length", "title"], axis=1, inplace=True)

# Filling in empty slots with mean of total number of credit lines currrently in the borrower's credit file
df["mort_acc"].value_counts()
acc_avg = df.groupby("total_acc").mean()["mort_acc"]

def fill_na(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_na(x['total_acc'], x['mort_acc']), axis=1)

df = df.dropna()

# Convert objects to integers
def converter(term):
    term = term.split()
    if term[0] == "36":
        return 36
    else:
        return 60
    
df["term"] = df["term"].apply(converter)

df.drop(["grade", "issue_d"], axis=1, inplace=True)

features_int = pd.get_dummies(df[['sub_grade', 'verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['sub_grade', 'verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,features_int],axis=1)

# Convert and replace object values
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
home_ownership_int = pd.get_dummies(df["home_ownership"], drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df, home_ownership_int],axis=1)

# Creating zip column from address
def zip_extract(address):
    zip = address.split()[-1]
    return zip

df["zip_code"] = df["address"].apply(zip_extract)

zip_code_int = pd.get_dummies(df["zip_code"], drop_first=True)
df.drop(["zip_code", "address"], axis=1, inplace=True)
df = pd.concat([df, zip_code_int], axis=1)


def convertInt(time):
    time = time.split("-")
    year = int(time[1])
    return year

df["earliest_cr_line"] = df["earliest_cr_line"].apply(convertInt)


### Train test split ###

df.drop("loan_status", axis=1, inplace=True)

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

# Grabbing a fraction of the entries to save time and compute on training.
df = df.sample(frac=0.1,random_state=101)
print(len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


### Creating model ### 
model = Sequential()

model.add(Dense(units=78,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=19,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))

# Save model
model.save('loan_model.h5')  


### Evaluate performance of model ###
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()

predictions = (model.predict(X_test) > 0.5).astype("int32")
confusion_matrix(y_test, predictions)
classification_report(y_test,predictions)

# Predict on new costumer
random_index = random.randint(0,len(df))
new_customer = df.drop('loan_repaid',axis=1).iloc[random_index]
new_customer = scaler.transform(new_customer.values.reshape(1,78))
(model.predict(new_customer) > 0.5).astype("int32")
df.iloc[random_index]["loan_repaid"]





