# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:25:18 2024
@author: Team 8 Code
"""

# Load Libraries
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

# Load Initial Datasets
branch_data = pd.read_excel('Branch_Level_Dataset.xlsx')
member_data = pd.read_csv('Member_Level_Dataset.csv')

# Check the Head and Size of Datasets
print(branch_data.shape)
print(member_data.shape)

# First, let's check for any missing values in the datasets
branch_missing_values = branch_data.isnull().sum()
member_missing_values = member_data.isnull().sum()


# Compute summary statistics for the numerical columns of the datasets
# Set option to display all columns
pd.set_option('display.max_columns', None)

# Now, when you call describe(), it should show statistics for all columns
branch_summary_statistics = branch_data.describe()
print(branch_summary_statistics)
branch_summary_statistics =branch_data.describe()
member_summary_statistics =member_data.describe()


# Exploratory Analysis of the given datasets:
    
# To advise on a better location, we could look at the total transactions per branch, which might indicate popularity or busyness.
# Aggregating the data to see the total transactions per branch category.
# We'll create a new column 'Total_Transactions' which is a sum of all transaction types.
branch_data['Total_Transactions'] = branch_data.iloc[:, 2:-2].sum(axis=1)

# Now, we group by BranchCategory and sum up the total transactions for each branch.
branch_transactions = branch_data.groupby('BranchCategory')['Total_Transactions'].sum().sort_values(ascending=False)
branch_transactions

# Next, we could look at trends over time by month.
# For this, we'll group by EOM_TRANS_DATE and sum the transactions to see the overall monthly trend.
monthly_transactions =branch_data.groupby('EOM_TRANS_DATE')['Total_Transactions'].sum()


# We also might want to consider the county-level data, so let's get the total transactions per county.
county_transactions = branch_data.groupby('County')['Total_Transactions'].sum().sort_values(ascending=False)

branch_missing_values,member_missing_values, branch_summary_statistics,member_summary_statistics, branch_transactions, monthly_transactions, county_transactions

# Generate a list of distinct colors for the top 10 branches
top_branches = branch_transactions.head(10)
branch_colors = plt.cm.get_cmap('tab20', len(top_branches))

# Set up the visualizations
plt.figure(figsize=(18, 12))

# Plot 1: Top Branches by Total Transactions with Different Colors for Each Bar
plt.subplot(2, 2, 2)  # Adjusting subplot for a single visualization

branches = top_branches.index
transactions = top_branches.values
colors = [branch_colors(i) for i in range(len(branches))]

# Using horizontal bars ('barh' instead of 'bar')
plt.barh(branches, transactions, color=colors)
plt.title('Top 10 Branches by Total Transactions')
plt.xlabel('Total Transactions')
plt.ylabel('Branch Category')
# Removing grid lines from the background
plt.grid(False)
plt.tight_layout()  # Adjust layout to fit everything
# Save the plot as a PNG file
plt.savefig(r'top_branches_plot.png', bbox_inches='tight', dpi=300)
plt.show()





# Get a list of all the unique counties in the dataset
counties = branch_data['County'].unique()

# Create a dictionary of dataframes, one for each county, with the aggregated total transactions per branch
county_data = {}
for county in counties:
    county_data[county] = branch_data[branch_data['County'] == county].groupby('BranchCategory')['Total_Transactions'].sum()

# Determine the number of rows needed for subplots (2 plots per row)
num_rows = (len(counties) + 1) // 2

# Set the figure size
plt.figure(figsize=(15, num_rows * 5))

# Create a subplot for each county
for i, (county, transactions) in enumerate(county_data.items(), 1):
    plt.subplot(num_rows, 2, i)  # Adjust for 2 plots per row
    
    # Generate a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(transactions)))

    transactions.plot(kind='barh', color=colors)
    plt.title(f'Total Transactions for {county} County')
    plt.ylabel('Branch Category')
    plt.xlabel('Total Transactions')
    plt.grid(False)
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust these values as needed for horizontal (hspace) and vertical (wspace) spacing    

plt.savefig(r'Total Transactions for each County.png', bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()


# Create a color map for the counties
counties = county_transactions.index.unique()
county_colors = plt.cm.get_cmap('tab20', len(counties))

# Plotting Total Transactions by County with Different Colors
plt.figure(figsize=(10, 6))
plt.subplot(1, 1, 1)
for i, county in enumerate(counties):
    plt.barh(county, county_transactions.loc[county], color=county_colors(i))

plt.title('Total Transactions')
plt.xlabel('Total Transactions by County')
plt.ylabel('County')
plt.xticks(rotation=45, ha='right')

plt.grid(False)
plt.savefig(r'Total Transactions by County.png', bbox_inches='tight', dpi=300)
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()

# Analyzing the geographical distribution of members
zip_code_counts = member_data['address_zip'].value_counts()

# Plotting the top 10 zip codes by number of members with different colors for each bar
plt.figure(figsize=(10, 6))
sns.barplot(y=zip_code_counts.head(10).index, 
            x=zip_code_counts.head(10).values, 
            palette=sns.color_palette("hsv", 10))  # Using a hue-saturation-value palette
plt.title('Top 10 Zip Codes by Number of Members')
plt.ylabel('Zip Code')
plt.xlabel('Number of Members')
plt.xticks(rotation=45)
plt.grid(False)
plt.savefig(r'Top 10 Zip Codes by Number of Members.png', bbox_inches='tight', dpi=300)
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()




# Calculating Total Transactions for each member
transaction_columns = ['ATMCount', 'BillPaymentCount', 'CashCount', 'DraftCount', 'ACHCount', 
                       'FeeCount', 'Credit_DebitCount', 'Home_Banking', 'WireCount', 'DividendCount']
member_data['Total_Transactions'] = member_data[transaction_columns].sum(axis=1)

# Grouping by zip code to analyze average age and total transaction volume
zip_code_financial_analysis = member_data.groupby('address_zip').agg(
    Average_Age=('age', 'mean'),
    Total_Transactions=('Total_Transactions', 'sum')
).reset_index()

# Sorting the results for top 10 zip codes by transactions
top_zip_codes_by_transactions_financial = zip_code_financial_analysis.sort_values(
    by='Total_Transactions', ascending=False
).head(10)

# Displaying the age distribution along with financial analysis
print("Age Distribution and Financial Analysis by Zip Code:")
print(top_zip_codes_by_transactions_financial)

# Grouping by zip code to analyze average age and total transaction volume
zip_code_demographic_analysis = member_data.groupby('address_zip').agg(
    Average_Age=('age', 'mean'),
    Total_Transactions=('Total_Transactions', 'sum')
).reset_index()

# Sorting the results for top 10 zip codes by average age and transactions
top_zip_codes_by_age = zip_code_demographic_analysis.sort_values(by='Average_Age', ascending=False).head(10)
top_zip_codes_by_transactions_demographic = zip_code_demographic_analysis.sort_values(by='Total_Transactions', ascending=False).head(10)
top_zip_codes_by_age, top_zip_codes_by_transactions_demographic





# Transaction types for the distribution graph
transaction_types = ['ATM', 'Bill Payment', 'Cash', 'Draft', 'ACH', 'Fee', 'Credit/Debit Card', 'Home Banking', 'Dividend']

# Plotting the distribution of the transaction types for the branch dataset
plt.figure(figsize=(14, 7))

# Since we want to compare the distributions, a boxplot will be useful
sns.boxplot(data=branch_data[transaction_types])
plt.title('Distribution of Transaction Types in Branch Dataset')
plt.ylabel('Transaction Counts')
plt.xlabel('Transaction Types')
plt.xticks(rotation=45)
plt.yscale('log')  # Using a log scale due to large variances in the data
plt.grid(False)
plt.savefig(r'Distribution of Transaction Types in Branch Dataset.png', bbox_inches='tight', dpi=300)
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()


# To see which transaction type has the highest counts, we can sum up each type and then compare them.
# We will create a bar chart to visualize the total counts for each transaction type.

# Define your custom color palette as a list of HEX codes or named colors
custom_palette = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

# Summing up the transaction counts for each type
total_transactions = branch_data[transaction_types].sum().sort_values(ascending=False)

# Plotting the total transaction counts for each type with the custom color palette
plt.figure(figsize=(12, 6))
sns.barplot(x=total_transactions.index, y=total_transactions.values, palette=custom_palette)
plt.title('Total Counts for Each Transaction Type')
plt.ylabel('Total Transaction Count')
plt.xlabel('Transaction Type')
plt.xticks(rotation=45)
plt.grid(False)
plt.savefig(r'Total Counts for Each Transaction Type.png', bbox_inches='tight', dpi=300)
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()


# We will create a series of bar charts, one for each transaction type, that show the number of transactions per branch.

# Setting up the figure size and grid for the bar charts
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.ravel()  # Flatten the array of axes for easier iteration

# Looping through each transaction type and creating a bar chart for it
for i, transaction_type in enumerate(transaction_types):
    # Get the top 10 branches for each transaction type for readability
    top_branches = branch_data.groupby('BranchCategory')[transaction_type].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_branches.values, y=top_branches.index, ax=axes[i])
    axes[i].set_title(f'Branches per {transaction_type} Transactions')
    axes[i].set_xlabel('Transaction Count')
    axes[i].set_ylabel('Branch')
    axes[i].grid(False)

# Adjust spacing between the subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust these values as needed for horizontal (hspace) and vertical (wspace) spacing
plt.savefig(r'Branches per {transaction_type} Transactions.png', bbox_inches='tight', dpi=300)
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()

# Model (Classification) Analyis
# Unique Branch Categories in Datasets
unique_branch_categories_branch_level = branch_data['BranchCategory'].unique()
unique_branch_categories_member_level = member_data['BranchCategory'].unique()
print("Unique Branch Categories in Branch Level Data:", unique_branch_categories_branch_level)
print("Unique Branch Categories in Member Level Data:", unique_branch_categories_member_level)

# Find Common and Unique Categories in Both Datasets
common_categories = set(unique_branch_categories_branch_level).intersection(unique_branch_categories_member_level)
unique_to_branch_level = set(unique_branch_categories_branch_level).difference(unique_branch_categories_member_level)
unique_to_member_level = set(unique_branch_categories_member_level).difference(unique_branch_categories_branch_level)
print("Common Branch Categories:", common_categories)
print("Unique to Branch Level Data:", unique_to_branch_level)
print("Unique to Member Level Data:", unique_to_member_level)

# Data Types and Missing Values
print(branch_data.info())
print(member_data.info())

# Summary Statistics
print(branch_data.describe())
print(member_data.describe())

# Aggregate Member Data
aggregated_member_data = member_data.groupby(['EOM_TRANS_DATE', 'BranchCategory']).agg({
    'Unique_Member_Identifier':'count',
    'age': 'mean',
    'address_zip': lambda x: x.mode()[0] if not x.mode().empty else np.NaN,
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

# Save Aggregated Member Data
aggregated_member_data.to_csv('aggregated_member_data_updated.csv', index=False)



# Summary statistics
summary =branch_level_updated.groupby('BranchCategory')['Success'].value_counts(normalize=True).unstack().fillna(0)
summary
# Visualization of the branch success score
# Bar chart for Success and Failure counts in each Branch Category
plt.figure(figsize=(15, 8))
sns.countplot(data=branch_level_updated, x='BranchCategory', hue='Success')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.title('Success and Failure Counts per Branch Category',fontsize=24)
plt.ylabel('Count',fontsize=16)
plt.xlabel('Branch Category',fontsize=16)
plt.legend(title='Success', labels=['Failure', 'Success'], title_fontsize=20,bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=18)


# Display the summary and the plot
plt.tight_layout()
plt.show()
# Get features from branch level data
X=branch_level_updated.drop(['EOM_TRANS_DATE', 'BranchCategory','Success'], axis=1)

X.shape
#Get Target(response) varaible
y=branch_level_updated["Success"]
len(y)
y
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
  

# Function to plot confusion matrix
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

#Qualitative Analysis of Supplumentry  Datasets
#1-plot Demographic and Educational Trends in St. Cloud (2018-2022) in St.Cloud
# Paths to the data files
file_population = 'St. Cloud_population.csv'
file_education = 'St. Cloud_EDUCATIONAL ATTAINMENT.csv'

# Reading the data from the files
df_population = pd.read_csv(file_population)
df_education = pd.read_csv(file_education)

# Column names for the years
years = ['Estimate_2018', 'Estimate_2019', 'Estimate_2020', 'Estimate_2021', 'Estimate_2022']

# Extracting total population and converting to numeric values
total_population_actual = [float(value.replace(',', '')) for value in df_population.iloc[0, 1:].values]

# Extracting educational attainment data
education_levels_actual = {
    'High School Graduate': df_education.loc[df_education['Label (Grouping)'].str.contains('High school graduate', na=False), years].values[0],
    'Bachelor Degree': df_education.loc[df_education['Label (Grouping)'].str.contains("Bachelor's degree", na=False), years].values[0],
    'Graduate Degree': df_education.loc[df_education['Label (Grouping)'].str.contains('Graduate or professional degree', na=False), years].values[0]
}

# Converting percentage strings to float values
for key in education_levels_actual.keys():
    education_levels_actual[key] = [float(val.strip('%')) if isinstance(val, str) else val for val in education_levels_actual[key]]

# Creating the chart
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line plot for total population
ax2 = ax1.twinx()
ax2.plot(years, total_population_actual, 'b-o', label='Total Population Trend', linewidth=2, markersize=8)
ax2.set_ylabel('Total Population', color='b')
ax2.tick_params('y', colors='b')

# Stacked bar chart for educational attainment
bottom_values = np.zeros(len(years))  # Initialize bottom values for stacking
colors = ['green', 'purple', 'orange']  # Colors for the stacked bars
for i, (education_level, percentages) in enumerate(education_levels_actual.items()):
    ax1.bar(years, percentages, label=education_level, bottom=bottom_values, color=colors[i], alpha=0.7)
    bottom_values += np.array(percentages)
    

ax1.set_xlabel('Year')
ax1.set_ylabel('Percentage (%)')
ax1.grid(False)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax2.grid(False)
plt.title('Demographic and Educational Trends in St. Cloud (2018-2022)')

plt.savefig(r'Demographic and Educational Trends in St. Cloud (2018-2022).png', bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()

#2-plot Demographic and Educational Trends in St. Cloud (2018-2022) in K issimmee
# Paths to the data files
file_population = 'kissimmee_population.csv'
file_education = 'kissimmee_EDUCATIONAL ATTAINMENT.csv'

# Reading the data from the files
df_population = pd.read_csv(file_population)
df_education = pd.read_csv(file_education)

# Column names for the years
years = ['Estimate_2018', 'Estimate_2019', 'Estimate_2020', 'Estimate_2021', 'Estimate_2022']

# Extracting total population and converting to numeric values
total_population_actual = [float(value.replace(',', '')) for value in df_population.iloc[0, 1:].values]

# Extracting educational attainment data
education_levels_actual = {
    'High School Graduate': df_education.loc[df_education['Label (Grouping)'].str.contains('High school graduate', na=False), years].values[0],
    'Bachelor Degree': df_education.loc[df_education['Label (Grouping)'].str.contains("Bachelor's degree", na=False), years].values[0],
    'Graduate Degree': df_education.loc[df_education['Label (Grouping)'].str.contains('Graduate or professional degree', na=False), years].values[0]
}

# Converting percentage strings to float values
for key in education_levels_actual.keys():
    education_levels_actual[key] = [float(val.strip('%')) if isinstance(val, str) else val for val in education_levels_actual[key]]

# Creating the chart
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line plot for total population
ax2 = ax1.twinx()
ax2.plot(years, total_population_actual, 'b-o', label='Total Population Trend', linewidth=2, markersize=8)
ax2.set_ylabel('Total Population', color='b')
ax2.tick_params('y', colors='b')

# Stacked bar chart for educational attainment
bottom_values = np.zeros(len(years))  # Initialize bottom values for stacking
colors = ['green', 'purple', 'orange']  # Colors for the stacked bars
for i, (education_level, percentages) in enumerate(education_levels_actual.items()):
    ax1.bar(years, percentages, label=education_level, bottom=bottom_values, color=colors[i], alpha=0.7)
    bottom_values += np.array(percentages)

ax1.set_xlabel('Year')
ax1.set_ylabel('Percentage (%)')
ax1.grid(False)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax2.grid(False)

plt.title('Demographic and Educational Trends in Kissimmee (2018-2022)')
plt.tight_layout()
plt.savefig(r'Demographic and Educational Trends in Kissimmee (2018-2022).png', bbox_inches='tight', dpi=300)

plt.show()

#3-plot Economic and Social Trends in St. Cloud
# Assuming the file paths for income, poverty, and marital status datasets
file_income = 'St. Cloud_INCOME.csv'
file_poverty = 'St. Cloud_POVERTY .csv'
file_marital_status = 'St. Cloud_MARITAL STATUS.csv'

# Reading the data
df_income = pd.read_csv(file_income)
df_poverty = pd.read_csv(file_poverty)
df_marital_status = pd.read_csv(file_marital_status)

# Extracting median income data
median_income_actual = df_income.iloc[0, 1:].values
median_income_actual = [float(value.replace(',', '')) for value in median_income_actual]

# Extracting poverty data - assuming a specific row contains the relevant data
poverty_level_actual = df_poverty.loc[df_poverty['Label (Grouping)'].str.contains('Below 100 percent of the poverty level', na=False), years].values[0]
poverty_level_actual = [float(val.strip('%')) for val in poverty_level_actual]

# Extracting marital status data - focusing on the married percentage as an example
married_percentage_actual = df_marital_status.loc[df_marital_status['Label (Grouping)'].str.contains('Now married, except separated', na=False), years].values[0]
married_percentage_actual = [float(val.strip('%')) for val in married_percentage_actual]

# Creating Chart 2 with actual data
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line plot for median income
ax1.plot(years, median_income_actual, 'r-o', label='Median Income Trend', linewidth=2, markersize=8)
ax1.set_ylabel('Median Income ($)', color='r')
ax1.tick_params('y', colors='r')
ax1.set_xlabel('Year')
ax1.grid(False)
# Bar plot for poverty level and married percentage on a secondary axis
ax2 = ax1.twinx()
ax2.bar(years, poverty_level_actual, alpha=0.5, color='blue', label='Poverty Levels')
ax2.bar(years, married_percentage_actual, alpha=0.5, color='pink', label='Married Percentage')
ax2.set_ylabel('Percentage (%)')
ax2.grid(False)
# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Economic and Social Trends in St. Cloud (2018-2022)')
plt.tight_layout()
plt.tight_layout()
plt.savefig(r'Economic and Social Trends in St. Cloud (2018-2022).png', bbox_inches='tight', dpi=300)
plt.show()

#4-plot Economic and Social Trends in Kissimmee
# Assuming the file paths for income, poverty, and marital status datasets
file_income = 'kissimmee_INCOME.csv'
file_poverty = 'kissimmee_POVERTY.csv'
file_marital_status = 'kissimmee_MARITAL STATUS.csv'

# Reading the data
df_income = pd.read_csv(file_income)
df_poverty = pd.read_csv(file_poverty)
df_marital_status = pd.read_csv(file_marital_status)

# Extracting median income data
median_income_actual = df_income.iloc[0, 1:].values
median_income_actual = [float(value.replace(',', '')) for value in median_income_actual]

# Extracting poverty data - assuming a specific row contains the relevant data
poverty_level_actual = df_poverty.loc[df_poverty['Label (Grouping)'].str.contains('Below 100 percent of the poverty level', na=False), years].values[0]
poverty_level_actual = [float(val.strip('%')) for val in poverty_level_actual]

# Extracting marital status data - focusing on the married percentage as an example
married_percentage_actual = df_marital_status.loc[df_marital_status['Label (Grouping)'].str.contains('Now married, except separated', na=False), years].values[0]
married_percentage_actual = [float(val.strip('%')) for val in married_percentage_actual]

# Creating Chart 2 with actual data
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line plot for median income
ax1.plot(years, median_income_actual, 'r-o', label='Median Income Trend', linewidth=2, markersize=8)
ax1.set_ylabel('Median Income ($)', color='r')
ax1.tick_params('y', colors='r')
ax1.set_xlabel('Year')
ax1.grid(False)
# Bar plot for poverty level and married percentage on a secondary axis
ax2 = ax1.twinx()
ax2.bar(years, poverty_level_actual, alpha=0.5, color='blue', label='Poverty Levels')
ax2.bar(years, married_percentage_actual, alpha=0.5, color='pink', label='Married Percentage')
ax2.set_ylabel('Percentage (%)')

# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax2.grid(False)
plt.title('Economic and Social Trends in Kissimmee (2018-2022)')
plt.tight_layout()
plt.savefig(r'Economic and Social Trends in Kissimmee (2018-2022).png', bbox_inches='tight', dpi=300)
plt.show()


# 4- Economic Status and Wealth Trends Zipcode 34769 (St.Cloud)



file_paths = {
    "Income": r"Zip34769_INCOME.csv",
    "Poverty": r"zipcode34769\Zip34769_POVERTY.csv"
}

# Loading the datasets
income_data = pd.read_csv(file_paths["Income"])
poverty_data = pd.read_csv(file_paths["Poverty"])

# Extracting relevant data for the charts
# Assuming we are interested in the first row of each dataset for this example
median_income_values = income_data.loc[0, ['Estimate_2018', 'Estimate_2019', 'Estimate_2020', 'Estimate_2021', 'Estimate_2022']]
poverty_rate_values = poverty_data.loc[1, ['Estimate_2018', 'Estimate_2019', 'Estimate_2020', 'Estimate_2021', 'Estimate_2022']]

# Converting the values to float for plotting
median_income_values = median_income_values.str.replace(',', '').astype(float)
poverty_rate_values = poverty_rate_values.str.replace('%', '').astype(float)

# Choosing new colors for the chart
income_bar_color = 'green'  # New color for Median Household Income bars
poverty_line_color = 'orange'  # Keeping the same color for Poverty Rate line

# Setting a thicker line width for the Poverty Rate line chart
line_width = 2.5  # Increasing the line width

plt.figure(figsize=(12, 6))

# Creating a bar chart for Median Household Income
barplot = sns.barplot(x=years, y=median_income_values, color=income_bar_color)

# Setting up the primary y-axis for Median Income
plt.ylabel('Median Income ($)')
plt.xlabel('Year')
plt.title('Economic Status and Wealth Trends Zipcode 34769 (St.Cloud) ')
plt.grid(False)
# Adding values on top of each bar
for index, value in enumerate(median_income_values):
    barplot.text(index, value, f'{value:.0f}', color='black', ha="center", va="bottom")

# Adding a secondary y-axis for Poverty Rate using a thicker line chart
ax2 = plt.gca().twinx()
sns.lineplot(x=years, y=poverty_rate_values, marker='o', ax=ax2, color=poverty_line_color, linewidth=line_width, label='Poverty Rate (%)')
ax2.set_ylabel('Poverty Rate (%)')
ax2.grid(False)
# Adjusting the legend to include only one legend on the left side
plt.legend(loc='upper left')
plt.savefig(r'Economic Status and Wealth Trends Zipcode 34769 (St.Cloud).png', bbox_inches='tight', dpi=300)
plt.show()


# 5- Economic Status and Wealth Trends Zipcode 34772 (St.Cloud)
file_paths = {
    "Income": r"Zip34772_INCOME.csv",
    "Poverty": r"Zip34772_POVERTY.csv"
}

# Loading the datasets
income_data = pd.read_csv(file_paths["Income"])
poverty_data = pd.read_csv(file_paths["Poverty"])

# Extracting relevant data for the charts
# Assuming we are interested in the first row of each dataset for this example
median_income_values = income_data.loc[0, ['Estimate_2018', 'Estimate_2019', 'Estimate_2020', 'Estimate_2021', 'Estimate_2022']]
poverty_rate_values = poverty_data.loc[1, ['Estimate_2018', 'Estimate_2019', 'Estimate_2020', 'Estimate_2021', 'Estimate_2022']]

# Converting the values to float for plotting
median_income_values = median_income_values.str.replace(',', '').astype(float)
poverty_rate_values = poverty_rate_values.str.replace('%', '').astype(float)

# Choosing new colors for the chart
income_bar_color = 'green'  # New color for Median Household Income bars
poverty_line_color = 'orange'  # Keeping the same color for Poverty Rate line

# Setting a thicker line width for the Poverty Rate line chart
line_width = 2.5  # Increasing the line width

plt.figure(figsize=(12, 6))

# Creating a bar chart for Median Household Income
barplot = sns.barplot(x=years, y=median_income_values, color=income_bar_color)

# Setting up the primary y-axis for Median Income
plt.ylabel('Median Income ($)')
plt.xlabel('Year')
plt.title('Economic Status and Wealth Trends Zipcode 34772 (St.Cloud) ')
plt.grid(False)
# Adding values on top of each bar
for index, value in enumerate(median_income_values):
    barplot.text(index, value, f'{value:.0f}', color='black', ha="center", va="bottom")

# Adding a secondary y-axis for Poverty Rate using a thicker line chart
ax2 = plt.gca().twinx()
sns.lineplot(x=years, y=poverty_rate_values, marker='o', ax=ax2, color=poverty_line_color, linewidth=line_width, label='Poverty Rate (%)')
ax2.set_ylabel('Poverty Rate (%)')
ax2.grid(False)
# Adjusting the legend to include only one legend on the left side
plt.legend(loc='upper left')
plt.savefig(r'Economic Status and Wealth Trends Zipcode 34772 (St.Cloud).png', bbox_inches='tight', dpi=300)

plt.show()

# 6- Employment and Industry Trends Zipcode 34769 (St.Cloud)'
# Function to process data (convert string estimates to integers)
def process_data_corrected(df):
    for col in df.columns[1:]:
        df[col] = df[col].str.replace(',', '').str.extract('(\d+)').fillna(0).astype(int)
    return df

# Load the datasets
class_of_worker_data = pd.read_csv(r'34769\Zip34769_CLASS OF WORKER.csv')
updated_employment_data = pd.read_csv(r'34769\Zip34769_EMPLOYMENT.csv')

# Process the datasets
class_of_worker_data = process_data_corrected(class_of_worker_data)
updated_employment_data = process_data_corrected(updated_employment_data)

# Filter the updated employment data to include only "In labor force" and "Not in labor force" for each year
labor_force_data = updated_employment_data[updated_employment_data['Label (Grouping)'].str.contains('In labor force|Not in labor force')]

# Filter the class of worker data to exclude the total population row
class_of_worker_filtered = class_of_worker_data[class_of_worker_data['Label (Grouping)'].str.contains('Civilian employed population') == False]

# Color palette for bars and lines
colors_for_bars = ['skyblue', 'lightgreen']
new_colors_for_lines = ['red', 'gold', 'darkorchid', 'darkcyan']

# Find the maximum value in the labor force data to set the y-axis limit
max_labor_force_value = labor_force_data.iloc[:, 1:].max().max()

# Creating the plot
plt.figure(figsize=(15, 8))

# Plot "In labor force" and "Not in labor force" as stacked bars
for i, (label, color) in enumerate(zip(labor_force_data['Label (Grouping)'], colors_for_bars)):
    plt.bar(labor_force_data.columns[1:], labor_force_data.iloc[i, 1:], label=label.strip(), color=color)

# Overlay class of worker data as thicker lines with new colors
for i, (row, color) in enumerate(zip(class_of_worker_filtered.iterrows(), new_colors_for_lines)):
    plt.plot(class_of_worker_filtered.columns[1:], row[1][1:], marker='o', linewidth=3, label=row[1]['Label (Grouping)'].strip(), color=color)

# Set the y-axis limit to accommodate the highest value in the labor force data with a buffer
plt.ylim(0, max_labor_force_value + 1000)

# Adding labels, title, and legend
plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Employment and Industry Trends Zipcode 34769 (St.Cloud)')
plt.legend(loc='upper left')
plt.grid(False)
plt.savefig('Employment and Industry Trends Zipcode 34769 (St.Cloud).png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# 7-Employment and Industry Trends Zipcode 34772 (St.Cloud)
# Function to process data (convert string estimates to integers)
def process_data_corrected(df):
    for col in df.columns[1:]:
        df[col] = df[col].str.replace(',', '').str.extract('(\d+)').fillna(0).astype(int)
    return df

# Load the datasets
class_of_worker_data = pd.read_csv(r'Zip34772_CLASS OF WORKER.csv')
updated_employment_data = pd.read_csv(r'Zip34772_EMPLOYMENT.csv')

# Process the datasets
class_of_worker_data = process_data_corrected(class_of_worker_data)
updated_employment_data = process_data_corrected(updated_employment_data)

# Filter the updated employment data to include only "In labor force" and "Not in labor force" for each year
labor_force_data = updated_employment_data[updated_employment_data['Label (Grouping)'].str.contains('In labor force|Not in labor force')]

# Filter the class of worker data to exclude the total population row
class_of_worker_filtered = class_of_worker_data[class_of_worker_data['Label (Grouping)'].str.contains('Civilian employed population') == False]

# Color palette for bars and lines
colors_for_bars = ['skyblue', 'lightgreen']
new_colors_for_lines = ['red', 'gold', 'darkorchid', 'darkcyan']

# Find the maximum value in the labor force data to set the y-axis limit
max_labor_force_value = labor_force_data.iloc[:, 1:].max().max()

# Creating the plot
plt.figure(figsize=(15, 8))

# Plot "In labor force" and "Not in labor force" as stacked bars
for i, (label, color) in enumerate(zip(labor_force_data['Label (Grouping)'], colors_for_bars)):
    plt.bar(labor_force_data.columns[1:], labor_force_data.iloc[i, 1:], label=label.strip(), color=color)

# Overlay class of worker data as thicker lines with new colors
for i, (row, color) in enumerate(zip(class_of_worker_filtered.iterrows(), new_colors_for_lines)):
    plt.plot(class_of_worker_filtered.columns[1:], row[1][1:], marker='o', linewidth=3, label=row[1]['Label (Grouping)'].strip(), color=color)

# Set the y-axis limit to accommodate the highest value in the labor force data with a buffer
plt.ylim(0, max_labor_force_value + 1000)

# Adding labels, title, and legend
plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Employment and Industry Trends Zipcode 34772 (St.Cloud)')
plt.legend(loc='upper left')
plt.grid(False)
plt.savefig('Employment and Industry Trends Zipcode 34772 (St.Cloud)).png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# 8- Socio-Demographic Profile Zip Code 34769 (St.Cloud)


# Function to process data for graphing
def process_data_for_graph(df):
    for col in df.columns[1:]:
        if '%' in df[col].iloc[0]:
            df[col] = df[col].str.rstrip('%').astype(float)
        else:
            df[col] = pd.to_numeric(df[col].str.replace(',', '').str.extract('(\d+)')[0], errors='coerce')
    return df

# Load the datasets
population_data = pd.read_csv(r'Zip34769_population.csv')
educational_data = pd.read_csv(r'Zip34769_EDUCATIONAL.csv')

# Process the datasets
population_data = process_data_for_graph(population_data)
educational_data = process_data_for_graph(educational_data)

# Filter out the total population row from the population dataset
total_population_data = population_data[population_data['Label (Grouping)'].str.contains('Total population')]

# Filter out the total population row from the educational dataset
educational_filtered = educational_data[educational_data['Label (Grouping)'].str.contains('Population 25 years and over') == False]

# Set the color palette for the educational data
educational_colors = ['royalblue', 'green', 'red', 'purple', 'orange']

# Creating the graph
plt.figure(figsize=(15, 10))

# Creating a secondary y-axis for educational percentages
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plotting total population data as larger bars
bars = ax1.bar(total_population_data.columns[1:], total_population_data.iloc[0, 1:], color='navy', width=0.6, label='Total Population')

# Adding the numerical values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

# Plotting educational data as line charts
for i, row in educational_filtered.iterrows():
    color = educational_colors[i % len(educational_colors)]
    ax2.plot(educational_filtered.columns[1:], row[1:], marker='o', color=color, label=row['Label (Grouping)'].strip(), linewidth=2)

# Setting labels, titles, and legends
ax1.set_ylabel('Total Population Number')
ax1.set_xlabel('Year')
ax1.grid(False)
ax2.set_ylabel('Percentage of Educational Attainment')
plt.title('Socio-Demographic Profile Zip Code 34769 (St.Cloud)')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.85))
ax2.grid(False)
plt.savefig('Socio-Demographic Profile Zip Code 34769 (St.Cloud).png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# 9- Socio-Demographic Profile Zip Code 34769 (St.Cloud)
# Function to process data for graphing
def process_data_for_graph(df):
    for col in df.columns[1:]:
        if '%' in df[col].iloc[0]:
            df[col] = df[col].str.rstrip('%').astype(float)
        else:
            df[col] = pd.to_numeric(df[col].str.replace(',', '').str.extract('(\d+)')[0], errors='coerce')
    return df

# Load the datasets
population_data = pd.read_csv(r'Zip34772_population.csv')
educational_data = pd.read_csv(r'Zip34772_EDUCATIONAL.csv')

# Process the datasets
population_data = process_data_for_graph(population_data)
educational_data = process_data_for_graph(educational_data)

# Filter out the total population row from the population dataset
total_population_data = population_data[population_data['Label (Grouping)'].str.contains('Total population')]

# Filter out the total population row from the educational dataset
educational_filtered = educational_data[educational_data['Label (Grouping)'].str.contains('Population 25 years and over') == False]

# Set the color palette for the educational data
educational_colors = ['royalblue', 'green', 'red', 'purple', 'orange']

# Creating the graph
plt.figure(figsize=(15, 10))

# Creating a secondary y-axis for educational percentages
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plotting total population data as larger bars
bars = ax1.bar(total_population_data.columns[1:], total_population_data.iloc[0, 1:], color='navy', width=0.6, label='Total Population')

# Adding the numerical values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

# Plotting educational data as line charts
for i, row in educational_filtered.iterrows():
    color = educational_colors[i % len(educational_colors)]
    ax2.plot(educational_filtered.columns[1:], row[1:], marker='o', color=color, label=row['Label (Grouping)'].strip(), linewidth=2)

# Setting labels, titles, and legends
ax1.set_ylabel('Total Population Number')
ax1.set_xlabel('Year')
ax1.grid(False)
ax2.set_ylabel('Percentage of Educational Attainment')
plt.title('Socio-Demographic Profile Zip Code 34772 (St.Cloud)')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.85))
ax2.grid(False)
plt.savefig(r'Socio-Demographic Profile Zip Code 34772 (St.Cloud).png', bbox_inches='tight', dpi=300)
plt.tight_layout()
# Show the plot
plt.show()
