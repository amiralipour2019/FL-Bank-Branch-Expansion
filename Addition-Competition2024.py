# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:25:18 2024

@author: Amir Alipour
"""
# Load Librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


#Load initial datasets

branch_data = pd.read_csv('D:\Classes\Spring 2024\Competition 2023-2024\Branch_Level_Dataset.csv')
member_data = pd.read_csv('D:\Classes\Spring 2024\Competition 2023-2024\Member_Level_Dataset.csv')
#branch_data_2 = pd.read_excel('D:\Classes\Spring 2024\Competition 2023-2024\Branch_Level_Dataset-2.xlsx')
# check the head and size of datasets
branch_data.shape
member_data.shape

# Get unique BranchCategory values from the branch level data
unique_branch_categories_branch_level = branch_data['BranchCategory'].unique()

# Get unique BranchCategory values from the member level data
unique_branch_categories_member_level = member_data['BranchCategory'].unique()

# print unique categories
print("Unique Branch Categories in Branch Level Data:", unique_branch_categories_branch_level)
print("Unique Branch Categories in Member Level Data:", unique_branch_categories_member_level)

# Convert the arrays to sets
branch_level_categories = set(unique_branch_categories_branch_level)
member_level_categories = set(unique_branch_categories_member_level)

# Find common categories in both datasets
common_categories = branch_level_categories.intersection(member_level_categories)
common_categories
# Find categories unique to the branch level dataset
unique_to_branch_level = branch_level_categories.difference(member_level_categories)
unique_to_branch_level
# Find categories unique to the member level dataset
unique_to_member_level = member_level_categories.difference(branch_level_categories)
unique_to_member_level
# Print the results
print("Common Branch Categories:", common_categories)
print("Unique to Branch Level Data:", unique_to_branch_level)
print("Unique to Member Level Data:", unique_to_member_level)

# Check the first few rows
print(branch_data.head())
print(member_data.head())
print(member_data.iloc[:5, 1:5])
# Data types and missing values
print(branch_data.info())
print(member_data.info())

# Summary statistics
print(branch_data.describe())
print(member_data.describe())

# # List of transaction type columns
# transaction_columns = ['ATM', 'Bill Payment', 'Cash', 'Draft', 'ACH', 'Fee', 'Credit/Debit Card', 'Home Banking', 'Dividend']

# # Calculate descriptive statistics
# transaction_stats = branch_data[transaction_columns].describe()

# Calculate and add the mode for each transaction type
# transaction_mode = branch_data[transaction_columns].mode().iloc[0]
# transaction_stats.loc['mode'] = transaction_mode
# transaction_mode

# # Count the number of transactions per branch category
# transactions_per_category_count = branch_data.groupby('BranchCategory').count()
# transactions_per_category_count

# # Calculate the average transaction count per branch category
# avg_transactions_per_category = branch_data.groupby('BranchCategory').mean()

# avg_transactions_per_category

# print("Descriptive Statistics for Transaction Types:")
# print(transaction_stats)

# print("\nCount of Transactions per Branch Category:")
# print(transactions_per_category_count)

# print("\nAverage Transactions per Branch Category:")
# print(avg_transactions_per_category)

# # Sum the transactions for each branch category
# transaction_totals = branch_data.groupby('BranchCategory').sum()

# # Plotting the bar chart
# plt.figure(figsize=(10, 6))
# sns.barplot(x=transaction_totals.index, y=transaction_totals['ATM'])

# # Adding title and labels
# plt.title('Total ATM Transactions per Branch Category')
# plt.xlabel('Branch Category')
# plt.ylabel('Total ATM Transactions')

# # Show the plot
# plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
# plt.show()

# # Create a mapping from BranchCategory to shorter labels
# unique_branches = branch_data['BranchCategory'].unique()
# branch_mapping = {branch: f'B{i+1}' for i, branch in enumerate(unique_branches)}

# # Add a new column for the encoded labels
# branch_data['BranchCode'] = branch_data['BranchCategory'].map(branch_mapping)

# # Sum the transactions for each encoded branch label
# transaction_totals = branch_data.groupby('BranchCode').sum()


# # Plotting the bar chart with encoded branch names
# plt.figure(figsize=(12, 6))
# sns.barplot(x=transaction_totals.index, y=transaction_totals['ATM'])

# # Adding title and labels
# plt.title('Total ATM Transactions per Encoded Branch Category')
# plt.xlabel('Encoded Branch Category')
# plt.ylabel('Total ATM Transactions')

# # Show the plot
# plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
# plt.show()

# # Add a text box for the BranchCategory mapping
# textstr = '\n'.join([f'{code}: {name}' for name, code in branch_mapping.items()])
# plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
#                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.5))
# #### merge branch level and member level data
# # Assuming both datasets are loaded as branch_data and member_data
# merged_data = pd.merge(member_data, branch_data, on='BranchCategory', how='left')
# merged_data.shape

# merged_data2 = pd.merge(member_data, branch_data, on=['EOM_TRANS_DATE', 'BranchCategory'], how='left')

# # Grouping by 'BranchCategory' and aggregating data
# aggregated_member_data = member_data.groupby('BranchCategory').agg({
#     'age': 'mean',  # Average age of members
#     'transaction_count': 'sum',  # Total number of transactions
#     # Include other relevant aggregations
# })


# # member_data_2 = pd.read_csv('D:\Classes\Fall2023\Competition 2023-2024\Member_Level_Dataset-2.csv')

# # # check the head and size of datasets
# # branch_data.shape
# # member_data_2.shape

# # # Check the first few rows
# # print(branch_data.head())
# # print(member_data_2.head())
# # print(member_data.iloc[:5, 1:5])


# Aggregate member data
member_data.columns
aggregated_member_data = member_data.groupby(['EOM_TRANS_DATE', 'BranchCategory']).agg({
    'Unique_Member_Identifier':'count',
    'age': 'mean',  # Average age
    'address_zip': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,  # Most common zip code
    'n_accts': 'sum',
    'n_checking_accts': 'sum',
    'n_savings_accts': 'sum',
    'n_open_loans': 'sum',
    'n_open_cds': 'sum',
    'n_open_club_accts': 'sum',
    'n_open_credit_cards': 'sum',
    'ATMCount': 'sum',
    'BillPaymentCount': 'sum',
    'CashCount': 'sum',
    'DraftCount': 'sum',
    'ACHCount': 'sum',
    'FeeCount': 'sum',
    'Credit_DebitCount': 'sum',
    'Home_Banking': 'sum',
    'WireCount': 'sum',
    'DividendCount': 'sum'
}).reset_index()

aggregated_member_data.shape
aggregated_member_data.to_csv(r'D:\Classes\Spring 2024\Competition 2023-2024\aggregated_member_data_updated.csv', index=False)

aggregated_member_data=pd.read_csv(r'D:\Classes\Spring 2024\Competition 2023-2024\aggregated_member_data_updated.csv')

aggregated_member_data.shape
branch_data.shape
# aggregated_member_data.head

# # Get unique BranchCategory values from the aggregated member data
# unique_branch_categories_aggregated_member_data = aggregated_member_data['BranchCategory'].unique()
# unique_branch_categories_aggregated_member_data
# unique_branch_categories_aggregated = set(aggregated_member_data['BranchCategory'].unique())

# common_branch_categories = unique_branch_categories_aggregated.intersection(unique_branch_categories_member_level)
# common_branch_categories

# unique_to_member_level2 = branch_level_categories.difference(member_level_categories)
# unique_to_branch_level

# # Print or inspect the common categories
# print("Common Branch Categories:", common_branch_categories)

# # Check data types
# print(aggregated_member_data.dtypes)
# print(branch_data.dtypes)

# # Ensure that 'BranchCategory' is of the same data type in both DataFrames
# branch_data['BranchCategory'] = branch_data['BranchCategory'].astype(aggregated_member_data['BranchCategory'].dtype)

# # Step 2: Merge the Aggregated Member Data with Branch Data
# merged_data = pd.concat(aggregated_member_data, branch_data, on=['EOM_TRANS_DATE', 'BranchCategory'], how='left')

# # Now, merged_data contains the combined data
# print(merged_data.head())

# # Convert EOM_TRANS_DATE to datetime in aggregated_member_data
# aggregated_member_data['EOM_TRANS_DATE'] = pd.to_datetime(aggregated_member_data['EOM_TRANS_DATE'], errors='coerce')

# # After conversion, reattempt the merge
# merged_data = pd.merge(aggregated_member_data, branch_data, on=['EOM_TRANS_DATE', 'BranchCategory'], how='left')
# merged_data.head

# merged_data.shape

# merged_data.to_csv(r'D:\Classes\Fall2023\Competition 2023-2024\merged_initial_data.csv', index=False)


# #Load  supplementary data 
# supplementary_data=pd.read_excel("D:\Classes\Spring 2024\Competition 2023-2024\Transposed_Supplementary.xlsx")
# supplementary_data.shape
# supplementary_data.columns
# supplementary_data.info
# column_names_list = supplementary_data.columns.tolist()

# relevant_columns = [
#     'Population 16 years and over',
#     'In labor force',
#     'Civilian labor force',
#     'Employed',
#     'Unemployed',
#     'Median household income (dollars)',
#     'Per capita income (dollars)',
#     'Manufacturing',
#     'Retail trade',
#     'Information',
#     'Finance and insurance, and real estate and rental and leasing',
#     'With health insurance coverage',
#     'No health insurance coverage',
#     'With Food Stamp/SNAP benefits in the past 12 months',
#     'Less than $10,000',
#     # Add any additional columns here, ensuring correct names
# ]

# filtered_supplementary_data = supplementary_data[relevant_columns]

# #CReating Target
# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder



# # If BranchCategory is a feature and you have a different target, replace 'TargetVariable' with your actual target column
# # Ensure the target variable is numeric for regression, or categorical (as integers) for classification
# # For this example, let's assume we're doing classification and BranchCategory is the target

# # Encode the categorical target variable
# branch_data_encoded=pd.read_csv("D:\\Classes\\Spring 2024\\Competition 2023-2024\\branch_data_encoded.csv")
# branch_data_encoded.head
# label_encoder = LabelEncoder()
# branch_data_encoded['BranchCategory_encoded'] = label_encoder.fit_transform(branch_data_encoded['BranchCategory'])
# branch_data_encoded.to_csv(r'D:\Classes\Spring 2024\Competition 2023-2024\branch_data_encoded.csv', index=False)
# # Separate features and target
# branch_data_encoded.columns
# X = branch_data_encoded.drop(['BranchCategory_encoded'], axis=1)  # Features

# X
# #branch_data_encoded.drop('BranchCategory','BranchCategory_encoded', axis=1)  # Features
# y = branch_data_encoded['BranchCategory_encoded']  # Target variable
# y

# X.info()
# # # Convert all string columns to categorical integer labels
# # for column in X.select_dtypes(include=['object']).columns:
# #     X[column] = LabelEncoder().fit_transform(X[column])

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# X_train.shape
# y_train.shape
# X_test.shape
# y_test.shape

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# X_train_scaled
# X_test_scaled
# # Initialize XGBoost classifier
# model = xgb.XGBClassifier(objective='multi:softprob', random_state=42)

# # Train the model
# model.fit(X_train_scaled, y_train)

# # Predictions
# predictions = model.predict(X_test_scaled)
# predictions
# # Get the probabilities of predictions
# prediction_probabilities = model.predict_proba(X_test_scaled)


# from sklearn.metrics import roc_curve, auc

# # Assuming binary classification and your positive class is 1
# probabilities = prediction_probabilities[:, 1]

# # Calculate ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, probabilities)



# # Display or process these predictions and probabilities as needed
# # For example, to print the first few predictions and their probabilities:
# print("Predictions:", predictions[:5])
# print("Probabilities:", prediction_probabilities[:5])
# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f"Model Accuracy: {accuracy * 100.0}%")
# print(classification_report(y_test, predictions))

# # If you need to inverse transform the predicted labels back to the original encoding
# predicted_categories = label_encoder.inverse_transform(predictions)




# # Defining weights for each transaction type and dividend
# weights = {
#     'ATM': 1, 
#     'Bill Payment': 1, 
#     'Cash': 1, 
#     'Draft': 1, 
#     'ACH': 1, 
#     'Fee': 1, 
#     'Credit/Debit Card': 1, 
#     'Home Banking': 1, 
#     'Dividend': 2
# }
# branch_data_wVirtuals=pd.read_csv("D:\\Classes\\Spring 2024\\Competition 2023-2024\\branch_data_withoutVirtuals.csv")
# branch_data_wVirtuals.shape
# branch_data_wVirtuals.head
# # Calculating the weighted sum for each branch
# weighted_sum = branch_data_wVirtuals.drop([ 'BranchCategory'], axis=1).mul(weights).sum(axis=1)
# weighted_sum
# # Normalizing these weighted sums to create the 'Branch Success' column
# branch_data_wVirtuals['Branch_Success_Weighted'] = (weighted_sum - weighted_sum.min()) / (weighted_sum.max() - weighted_sum.min())

# # Displaying the updated dataset with the new weighted 'Branch Success' column
# branch_data_wVirtuals.head()
# branch_data_wVirtuals.to_csv(r'D:\\Classes\\Spring 2024\\Competition 2023-2024\\branch_data_wVirtual_weighted.csv', index=False)




# branch_data_wVirtual_weighted=pd.read_csv("D:\\Classes\\Spring 2024\\Competition 2023-2024\\branch_data_wVirtual_weighted.csv")
# branch_data_wVirtual_weighted.shape
# branch_data_wVirtual_weighted.columns
# # Plotting the distribution of 'Branch Success' scores
# plt.figure(figsize=(10, 6))
# plt.hist(branch_data_wVirtual_weighted['Branch_Success_Weighted'], bins=20, color='skyblue', edgecolor='black')
# plt.title('Distribution of Branch Success Scores')
# plt.xlabel('Branch Success Score')
# plt.ylabel('Number of Branches')
# plt.grid(True)
# plt.show()

# quantile_75 = branch_data_wVirtual_weighted['Branch_Success_Weighted'].quantile(0.75)
# quantile_75

# # Creating a new column 'Success_Flag' based on the 75th percentile threshold
# threshold = quantile_75
# branch_data_wVirtual_weighted['Success_Flag'] = (branch_data_wVirtual_weighted['Branch_Success_Weighted'] >= threshold).astype(int)
# branch_data_wVirtual_weighted
# branch_data_wVirtual_weighted.to_csv(r'D:\\Classes\\Spring 2024\\Competition 2023-2024\\branch_data_wVirtual_weighted.csv', index=False)

### Read updated branch level with branch success
branch_level_updated=pd.read_excel(r"D:\\Classes\\Spring 2024\\Competition 2023-2024\\Branch_Dataset_update.xlsx")
branch_level_updated.shape
branch_level_updated.columns

#pip install numpy scikit-learn xgboost


# Example dataset (replace with your own data)
X=branch_level_updated.drop(['EOM_TRANS_DATE', 'BranchCategory','Success'], axis=1)

X.shape
y=branch_level_updated["Success"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Random Forest': RandomForestClassifier()
}

# Training and evaluating each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    test_error_rate = 1 - accuracy

    print(f"{name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Test Error Rate: {test_error_rate}\n")
  




# # Model names
# models = ['Logistic Regression', 'SVM', 'XGBoost', 'Random Forest']

# # Metrics
# accuracy = [0.993006993006993, 0.986013986013986, 0.993006993006993, 1.0]
# precision = [1.0, 1.0, 0.975609756097561, 1.0]
# test_error_rate = [0.006993006993006978, 0.013986013986013957, 0.006993006993006978, 0.0]

# # Colors for each model
# colors = ['skyblue', 'lightgreen', 'salmon', 'violet']

# # Creating subplots
# fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# # Plotting Accuracy
# axes[0].barh(models, accuracy, color=colors)
# axes[0].set_title('Accuracy')
# axes[0].set_xlim([0.95, 1.05])
# for i, v in enumerate(accuracy):
#     axes[0].text(v + 0.002, i, "{:.2f}".format(v), va='center')

# # Plotting Precision
# axes[1].barh(models, precision, color=colors)
# axes[1].set_title('Precision')
# axes[1].set_xlim([0.95, 1.05])
# for i, v in enumerate(precision):
#     axes[1].text(v + 0.002, i, "{:.2f}".format(v), va='center')

# # Plotting Test Error Rate
# axes[2].barh(models, test_error_rate, color=colors)
# axes[2].set_title('Test Error Rate')
# axes[2].set_xlim([0, 0.02])
# for i, v in enumerate(test_error_rate):
#     axes[2].text(v + 0.001, i, "{:.4f}".format(v), va='center')

# # Layout adjustments
# plt.tight_layout()
# plt.show()

# # Predict on training and test sets
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)



# # Calculate accuracies
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# # Check for overfitting
# is_overfitting = train_accuracy > test_accuracy

# # Visualization
# plt.figure(figsize=(10, 6))
# plt.bar(['Training Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'green'])
# plt.ylabel('Accuracy')
# plt.title('Model Performance: Training vs Test')
# plt.text(0, train_accuracy - 0.05, f'{train_accuracy:.2f}', ha='center', va='bottom', color='white')
# plt.text(1, test_accuracy - 0.05, f'{test_accuracy:.2f}', ha='center', va='bottom', color='white')

# if is_overfitting:
#     plt.suptitle('The model is overfitting', color='red')
# else:
#     plt.suptitle('The model is not overfitting', color='green')

# plt.show()

# import matplotlib.pyplot as plt

# # Model names
# models = ['Logistic Regression', 'SVM', 'XGBoost', 'Random Forest']

# # Metrics
# accuracy = [0.993, 0.986, 0.993, 1.000]
# precision = [1.000, 1.000, 0.976, 1.000]
# test_error_rate = [0.007, 0.014, 0.007, 0.000]

# # Creating subplots
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Plotting Accuracy
# axes[0].bar(models, accuracy, color='blue')
# axes[0].set_title('Accuracy')
# axes[0].set_ylim([0.95, 1.05])
# for i, v in enumerate(accuracy):
#     axes[0].text(i, v + 0.002, "{:.3f}".format(v), ha='center')

# # Plotting Precision
# axes[1].bar(models, precision, color='green')
# axes[1].set_title('Precision')
# axes[1].set_ylim([0.95, 1.05])
# for i, v in enumerate(precision):
#     axes[1].text(i, v + 0.002, "{:.3f}".format(v), ha='center')

# # Plotting Test Error Rate
# axes[2].bar(models, test_error_rate, color='red')
# axes[2].set_title('Test Error Rate')
# axes[2].set_ylim([0, 0.02])
# for i, v in enumerate(test_error_rate):
#     axes[2].text(i, v + 0.001, "{:.3f}".format(v), ha='center')

# # Layout adjustments
# plt.tight_layout()
# plt.show()


# Assuming X and y are your features and target variable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features (assuming this is necessary for your dataset)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Initializing models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Random Forest': RandomForestClassifier()
}

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name, ax, title='Confusion Matrix', cmap=plt.cm.Blues):
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_title(f'{model_name} - {title}')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

# Plotting confusion matrices for each model
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Computing the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plotting the confusion matrix
    plot_confusion_matrix(conf_matrix, name, ax)

plt.tight_layout()
plt.show()


def plot_confusion_matrix(cm, model_name, ax, cmap, title='Confusion Matrix'):
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_title(f'{model_name}')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


# Assign different color maps to each model
color_maps = {
    'Logistic Regression': 'Blues',
    'SVM': 'Greens',
    'XGBoost': 'Oranges',
    'Random Forest': 'Purples'
}

# Plotting confusion matrices with different color maps for each model
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Computing the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plotting the confusion matrix with specified color map
    cmap = color_maps[name]
    plot_confusion_matrix(conf_matrix, name, ax, cmap=cmap)

plt.tight_layout()
plt.show()

aggregated_member_data.columns
aggregated_data_by_member = aggregated_member_data.groupby('BranchCategory')['Unique_Member_Identifier'].sum().reset_index()
aggregated_data_by_member
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 12))
colors =  plt.cm.tab20(np.linspace(0, 1, len(aggregated_data_by_member['BranchCategory'])))
plt.barh(aggregated_data_by_member['BranchCategory'],
         aggregated_data_by_member['Unique_Member_Identifier'], color=colors)
plt.xlabel('Branch Name')
plt.ylabel('Number of Members')
plt.title('Number of Members per Branch')
plt.xticks(rotation=45)  # This rotates the branch names for better visibility
plt.yticks(fontsize=8)
plt.show()
