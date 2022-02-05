import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/lending_club_loan_two.csv")

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



