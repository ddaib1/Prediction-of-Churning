### EDA - EXPLORATORY DATA ANALYSIS ###
## Importing necessary Libraries ##

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

## Importing and Studying Dataset ##

dataset = pd.read_csv('churn_data.csv') #Dataset Imported

# Viewing Details about Dataset
dataset.head() 
dataset.columns
dataset.describe()

## Data Cleaning ##

dataset[dataset.credit_score < 300]
dataset = dataset[dataset.credit_score >= 300]

dataset.isna().any()

dataset.isna().sum()

dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])

dataset2 = dataset.drop(columns = ['user', 'churn'])

## Data Visualization ##

#Selecting classifying features
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]

fig = plt.figure(figsize=(20, 12))

#Displaying pie charts of all the features in two halves
plt.suptitle('Pie Chart Distributions Pt. 1', fontsize=15)
for i in range(1, dataset2.shape[1]//2 + 2):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
   
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    #Normalize presents data in percentages
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal') #X and Y Axes are abstracted by this line
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.suptitle('Pie Chart Distributions Pt. 2', fontsize=15)
for i in range(dataset2.shape[1]//2 + 2, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i-(dataset2.shape[1]//2 + 1))
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
   
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

## Exploring the uneven features and strength of correlation ##

#Checking how much these factors affect final result
dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
dataset[dataset2.cancelled_loan == 1].churn.value_counts()
dataset[dataset2.received_loan == 1].churn.value_counts()
dataset[dataset2.rejected_loan == 1].churn.value_counts()
dataset[dataset2.left_for_one_month == 1].churn.value_counts()

#Plotting strength of correlation of each feature
dataset2.drop(columns = ['housing', 'payment_type',
                         'registered_phones', 'zodiac_sign']
    ).corrwith(dataset.churn).plot.bar(figsize=(15,10),
              title = 'Correlation with Response variable',
              fontsize = 10, rot = 30,
              grid = True)

## Generating Heat Map ##

#Creating the correlation matrix
corr = dataset.drop(columns = ['user', 'churn']).corr()

sn.set(style="white")

# Mask the upper half of correlation matrix due to redundancy
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Removing fields that are weakly correlated
dataset = dataset.drop(columns = ['app_web_user'])

dataset.to_csv('new_churn_data.csv', index = False)
