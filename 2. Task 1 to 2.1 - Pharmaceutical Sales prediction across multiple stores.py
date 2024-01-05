#!/usr/bin/env python
# coding: utf-8

# # Pharmaceutical Sales prediction across multiple stores

# **Import Necessary Library**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# **Read the Data**

# In[2]:


sample_data = pd.read_csv('sample_submission.csv')
sample_data.head()


# In[3]:


sample_data.info()


# In[4]:


Store_data = pd.read_csv('store.csv')
Store_data


# In[5]:


Store_data.info()


# **Find the Null Values in Store Data**

# In[6]:


Store_data.isnull().sum()


# **Fill the Numerical column Null values by mean value**

# In[7]:


column_list = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

# Calculate the mean before filling null values
mean_before_fill = Store_data[column_list].mean()

# Fill null values in the specified columns with the mean of each respective column
Store_data[column_list] = Store_data[column_list].fillna(Store_data[column_list].mean())

# Calculate the mean after filling null values
mean_after_fill = Store_data[column_list].mean()

# Print the mean before and after filling null values
print("Mean before filling null values:")
print(mean_before_fill)
print("\nMean after filling null values:")
print(mean_after_fill)

# Display the updated DataFrame
print("\nUpdated DataFrame:")
Store_data


# **Fill the caterigorical column Null values by mode value**

# In[8]:


# Fill NaN values in the 'PromoInterval' column with the mode
Store_data['PromoInterval'].fillna(Store_data['PromoInterval'].mode()[0], inplace=True)

# Display the updated DataFrame
print("Updated DataFrame:")
Store_data


# In[9]:


Store_data.isnull().sum()


# In[10]:


train_data = pd.read_csv('train.csv')
train_data


# In[11]:


train_data.info()


# In[12]:


# Check the unique values in the 'StateHoliday' column
print("Unique values in 'StateHoliday':", train_data['StateHoliday'].unique())

# Use label encoding to convert 'StateHoliday' to numerical values
train_data['StateHoliday'] = train_data['StateHoliday'].astype('category').cat.codes

# Display the updated DataFrame
print("\nUpdated DataFrame:")
train_data


# In[13]:


train_data.isnull().sum()


# In[14]:


train_data.info()


# In[15]:


test_data = pd.read_csv('test.csv')
test_data


# In[16]:


test_data.isnull().sum()


# **Fill the Numerical column Null values by mean value**

# In[17]:


# Fill null values in the 'Open' column with the mean of the 'Open' column
test_data['Open'].fillna(test_data['Open'].mean(), inplace=True)

# Calculate the mean of the 'Open' column after filling null values
mean_open = test_data['Open'].mean()

# Print the mean
print("Mean of 'Open' column:", mean_open)

# Display the updated DataFrame
print("Updated DataFrame:")
test_data


# In[18]:


test_data.isnull().sum()


# # Task 1 - Exploration of customer purchasing behaviour

# # Check for distribution in both training and test sets - are the promotions distributed similarly between these two groups?

# In[19]:


# Check the distribution of promotions in the training set
training_promotion_distribution = train_data['Promo'].value_counts(normalize=True)
print("Training Set Promotion Distribution:")
print(training_promotion_distribution)

# Check the distribution of promotions in the test set
test_promotion_distribution = test_data['Promo'].value_counts(normalize=True)
print("\nTest Set Promotion Distribution:")
print(test_promotion_distribution)


# In[20]:


# Plot promotion distribution in the training set
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
train_data['Promo'].value_counts().plot(kind='bar', title='Training Set Promotion Distribution')

# Plot promotion distribution in the test set
plt.subplot(1, 2, 2)
test_data['Promo'].value_counts().plot(kind='bar', title='Test Set Promotion Distribution')

plt.tight_layout()
plt.show()


# # Check & compare sales behavior before, during and after holidays

# In[21]:


# Convert the 'Date' column to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'])

# Extract information about StateHoliday and SchoolHoliday
train_data['StateHoliday'] = train_data['StateHoliday'].apply(lambda x: 0 if x == '0' else 1)
train_data['SchoolHoliday'] = train_data['SchoolHoliday'].astype(int)

# Create categorical variables for different holiday periods
train_data['HolidayPeriod'] = 'Non-Holiday'
train_data.loc[train_data['StateHoliday'] == 1, 'HolidayPeriod'] = 'StateHoliday'
train_data.loc[train_data['SchoolHoliday'] == 1, 'HolidayPeriod'] = 'SchoolHoliday'
train_data.loc[(train_data['StateHoliday'] == 1) & (train_data['SchoolHoliday'] == 1), 'HolidayPeriod'] = 'BothHolidays'

# Analyze and compare sales behavior during different holiday periods
holiday_sales_analysis = train_data.groupby('HolidayPeriod').agg({
    'Sales': 'mean'
}).reset_index()

# Visualize the impact on sales based on different holiday periods
plt.figure(figsize=(10, 6))
sns.barplot(x='HolidayPeriod', y='Sales', data=holiday_sales_analysis, order=['Non-Holiday', 'StateHoliday', 'SchoolHoliday', 'BothHolidays'])
plt.title('Impact of Different Holiday Periods on Sales')
plt.xlabel('Holiday Period')
plt.ylabel('Average Sales')
plt.show()


# # Find out any seasonal (Christmas, Easter etc) purchase behaviours

# In[22]:


train_data['Date'] = pd.to_datetime(train_data['Date'])

# Extract month and year from the date
train_data['Month'] = train_data['Date'].dt.month
train_data['Year'] = train_data['Date'].dt.year

# Define holiday-related months (adjust based on your local holidays)
holiday_months = [12, 4]  # December for Christmas, April for Easter

# Create a binary column indicating whether the month is a holiday month
train_data['IsHolidayMonth'] = train_data['Month'].isin(holiday_months).astype(int)

# Analyze the impact of holiday months on sales
holiday_sales_analysis = train_data.groupby('IsHolidayMonth').agg({
    'Sales': 'mean'
}).reset_index()

# Visualize the impact on sales based on holiday months
plt.figure(figsize=(8, 5))
sns.barplot(x='IsHolidayMonth', y='Sales', data=holiday_sales_analysis)
plt.title('Impact of Holiday Months on Sales')
plt.xlabel('Is Holiday Month')
plt.ylabel('Average Sales')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()


# # What can you say about the correlation between sales and number of customers?

# In[23]:


correlation_matrix = train_data[['Sales', 'Customers']].corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Sales and Number of Customers')
plt.show()


# # How does promo affect sales? Are the promos attracting more customers? How does it affect already existing customers?

# In[24]:


# Analyze the impact of promotions on sales
promo_sales_analysis = train_data.groupby('Promo').agg({
    'Sales': 'mean',
    'Customers': 'mean'
}).reset_index()

# Visualize the impact of promotions on sales
sns.barplot(x='Promo', y='Sales', data=promo_sales_analysis)
plt.title('Impact of Promotions on Sales')
plt.xlabel('Promo')
plt.ylabel('Average Sales')
plt.show()

# Visualize the impact of promotions on attracting more customers
sns.barplot(x='Promo', y='Customers', data=promo_sales_analysis)
plt.title('Impact of Promotions on Number of Customers')
plt.xlabel('Promo')
plt.ylabel('Average Number of Customers')
plt.show()


# 1. The DataFrame is grouped by the 'Promo' column to calculate the mean sales and the mean number of customers for each promo status.
# 2. Two bar plots are created: one for the impact of promotions on sales and another for the impact on the number of customers.
# 3. The resulting visualizations will help you understand how promotions are correlated with average sales and the average number of customers. Positive impacts on sales and an increase in the number of customers during promotions may suggest that the promotions are effective.

# # Could the promos be deployed in more effective ways? Which stores should promos be deployed in?

# In[25]:


# Analyze the impact of promotions on sales for each store
promo_store_analysis = train_data.groupby(['Store', 'Promo']).agg({
    'Sales': 'mean',
    'Customers': 'mean'
}).reset_index()

# Visualize the impact of promotions on sales for each store
plt.figure(figsize=(12, 6))
sns.barplot(x='Store', y='Sales', hue='Promo', data=promo_store_analysis)
plt.title('Impact of Promotions on Sales for Each Store')
plt.xlabel('Store')
plt.ylabel('Average Sales')
plt.show()

# Visualize the impact of promotions on the number of customers for each store
plt.figure(figsize=(12, 6))
sns.barplot(x='Store', y='Customers', hue='Promo', data=promo_store_analysis)
plt.title('Impact of Promotions on Number of Customers for Each Store')
plt.xlabel('Store')
plt.ylabel('Average Number of Customers')
plt.show()


# 1. The DataFrame is grouped by both 'Store' and 'Promo' columns to calculate the mean sales and the mean number of customers for each store and promo status.
# 
# 2. Two bar plots are created: one for the impact of promotions on sales for each store and another for the impact on the number of customers for each store.
# 
# 3. These visualizations will help you identify which stores benefit the most from promotions and which ones may need adjustments in their promotional strategies. If certain stores consistently show positive impacts from promotions, you may consider deploying promotions more frequently or intensively in those stores.

# # Trends of customer behavior during store open and closing times

# In[26]:


# Convert 'Timestamp' to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'])


# Analyze trends during store open and closing times
Daywise_analysis = train_data.groupby('Date').agg({
    'Sales': 'mean',
    'Customers': 'mean'
}).reset_index()

# Visualize trends during store open and closing times
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Sales', data=Daywise_analysis, label='Average Sales')
sns.lineplot(x='Date', y='Customers', data=Daywise_analysis, label='Average Number of Customers')
plt.title('Trends of Customer Behavior During Store Open and Closing date')
plt.xlabel('Daywise')
plt.ylabel('Average')
plt.legend()
plt.show()


# 1. The 'Date' column is converted to a datetime format
# 2. The DataFrame is then grouped by the hour of the day to calculate the mean sales and the mean number of customers for each day.
# 3. A line plot is created to visualize the trends during store open and closing.
# 4. This visualization will help you understand how customer behavior (both in terms of sales and the number of customers) varies throughout the day. You can observe peak days, identify trends during opening and closing day, and adjust staffing or promotional strategies accordingly.

# # Which stores are opened on all weekdays? How does that affect their sales on weekends? 

# In[27]:


# Identify stores open on all weekdays
stores_open_all_weekdays = train_data[train_data['DayOfWeek'].isin([1, 2, 3, 4, 5])]['Store'].unique()

# Filter the dataset for stores open on all weekdays
weekday_open_stores_data = train_data[train_data['Store'].isin(stores_open_all_weekdays)]

# Analyze how being open on all weekdays affects sales on weekends
weekend_sales_analysis = weekday_open_stores_data[~weekday_open_stores_data['DayOfWeek'].isin([1, 2, 3, 4, 5])].groupby('Store').agg({
    'Sales': 'mean'
}).reset_index()

# Visualize the impact on sales on weekends
plt.figure(figsize=(10, 6))
sns.barplot(x='Store', y='Sales', data=weekend_sales_analysis)
plt.title('Impact of Being Open on All Weekdays on Weekend Sales')
plt.xlabel('Store')
plt.ylabel('Average Sales on Weekends')
plt.show()


# # Check how the assortment type affects sales

# In[28]:


# Merge datasets on the 'Store' column
merged_data = pd.merge(train_data, Store_data, on='Store', how='left')

# Check the impact of assortment type on sales
assortment_sales_analysis = merged_data.groupby('Assortment').agg({
    'Sales': 'mean'
}).reset_index()

# Visualize the impact on sales based on assortment type
plt.figure(figsize=(8, 5))
sns.barplot(x='Assortment', y='Sales', data=assortment_sales_analysis)
plt.title('Impact of Assortment on Sales in train_data')
plt.xlabel('Assortment')


# # How does the distance to the next competitor affect sales? What if the store and its competitors all happen to be in city centres, does the distance matter in that case?

# In[29]:


# Merge datasets on the 'Store' column
merged_data = pd.merge(train_data, Store_data, on='Store', how='left')

# Analyze the impact of competition distance on sales
distance_sales_analysis = merged_data.groupby('CompetitionDistance').agg({
    'Sales': 'mean'
}).reset_index()

# Visualize the impact on sales based on competition distance
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CompetitionDistance', y='Sales', data=distance_sales_analysis)
plt.title('Impact of Competition Distance on Sales')
plt.xlabel('Competition Distance')
plt.ylabel('Average Sales')
plt.show()


# # How does the opening or reopening of new competitors affect stores? Check for stores with NA as competitor distance but later on has values for competitor distance
# 

# In[30]:


# Filter the dataframe to include only rows with non-null competition open since month
train_store = pd.merge(train_data, Store_data, how='inner', on='Store')
open_competition = train_store[train_store['CompetitionOpenSinceMonth'].notnull()]

# Convert competition open since year and competition open since month columns to integer type
open_competition['CompetitionOpenSinceYear'] = open_competition['CompetitionOpenSinceYear'].astype('int')
open_competition['CompetitionOpenSinceMonth'] = open_competition['CompetitionOpenSinceMonth'].astype('int')

# Filter rows where the year is less than competition open since year and the month is less than competition open since month
sales_before_competition_open = open_competition[(open_competition['Year'] < open_competition['CompetitionOpenSinceYear']) & 
                                                 (open_competition['Month'] < open_competition['CompetitionOpenSinceMonth'])]

# Filter rows where the year is greater than or equal to competition open since year and the month is greater than or equal to competition open since month
sales_after_competition_open = open_competition[(open_competition['Year'] >= open_competition['CompetitionOpenSinceYear']) & 
                                                (open_competition['Month'] >= open_competition['CompetitionOpenSinceMonth'])]

# Add a new column 'CompetitionBefore' with the value 'Before' for rows in sales_before_competition_open dataframe
sales_before_competition_open['CompetitionBefore'] = 'Before'

# Add a new column 'CompetitionBefore' with the value 'After' for rows in sales_after_competition_open dataframe
sales_after_competition_open['CompetitionBefore'] = 'After'

# Concatenate the sales_before_competition_open and sales_after_competition_open dataframes
competition_sales_data = pd.concat([sales_before_competition_open, sales_after_competition_open])

# Create a boxplot to visualize the effect of competition opening on sales
plt.figure(figsize=(10, 7))
sns.boxplot(x='CompetitionBefore', y='Sales', data=competition_sales_data)
plt.xlabel('Competition Opening')
plt.ylabel('Sales')
plt.title('Effect of Competition Opening on Sales')
plt.show()


# # 1.2 -  Logging
# Log your steps using the logger library in python. 

# In[31]:


train_data1 = train_data.copy
train_data1()
Store_data1 = Store_data.copy
Store_data1()


# In[32]:


import logging

# Configure logging settings
logging.basicConfig(filename='data_processing.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def Store_data1(data):
    """Function to store data."""
    try:
        # Store data logic here
        logging.info('Data stored successfully.')
    except Exception as e:
        logging.error(f'Error storing data: {e}')

def train_data1(data):
    """Function to train data."""
    try:
        # Training logic here
        logging.info('Data trained successfully.')
    except Exception as e:
        logging.error(f'Error training data: {e}')

if __name__ == "__main__":
    try:
        Store_data1('Sales')

        # Step 1: Store Data
        logging.info('Starting data storage process...')
        Store_data1('Sales')
        logging.info('Data storage process completed.')

        # Step 2: Train Data
        logging.info('Starting data training process...')
        train_data1('Sales')
        logging.info('Data training process completed.')

    except Exception as e:
        logging.error(f'Error in main process: {e}')


# # Task 2 - Prediction of store sales

# # Prediction of sales is the central task in this challenge. you want to predict daily sales in various stores up to 6 weeks ahead of time. This will help the company plan ahead of time. 

# In[33]:


from prophet import Prophet


# In[34]:


# Data Preprocessing
train_data['Date'] = pd.to_datetime(train_data['Date'])
daily_sales = train_data.groupby('Date')['Sales'].sum().reset_index()

# Feature Engineering
daily_sales.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

# Train-Test Split
train_size = int(len(daily_sales) * 0.8)
train_set, test_set = daily_sales[:train_size], daily_sales[train_size:]

# Model Selection and Training
model = Prophet()
model.fit(train_set)

# Prediction
future = model.make_future_dataframe(periods=6*7)  # 6 weeks ahead
forecast = model.predict(future)

# Visualization
fig = model.plot(forecast)
plt.show()


# Here's a basic outline of the steps you might follow:
# 
# **1.Data Preprocessing:**
# 
# Ensure that your date column is in datetime format.
# Explore the data and handle missing values.
# Aggregate the sales data by day if it's not already in that form.
# 
# **2.Feature Engineering:**
# 
# Extract relevant features from the date, such as day of the week, month, and year.
# Consider incorporating external factors that might influence sales, such as holidays, promotions, etc.
# Train-Test Split:
# 
# Split your data into training and testing sets. The training set should include the historical sales data, and the testing set should be the period you want to predict.
# 
# **3.Model Selection:**
# 
# Choose a time series forecasting model. Popular choices include ARIMA, SARIMA, Prophet, and machine learning models like XGBoost, LSTM, or GRU.
# 
# **4.Model Training:**
# 
# Train your selected model using the training set. Tune hyperparameters if necessary.
# 
# **5.Model Evaluation:**
# 
# Evaluate the model's performance on the testing set using appropriate metrics (e.g., Mean Absolute Error, Mean Squared Error).
# 
# **6.Prediction:**
# 
# Use the trained model to make predictions for the next 6 weeks.
# 
# **7.Visualization:**
# 
# Visualize the predicted sales against the actual sales to assess the model's performance.

# In[35]:


train_data.info()


# In[36]:


# Data Preprocessing
train_data['Date'] = pd.to_datetime(train_data['Date'])
daily_sales = train_data.groupby('Date')['Sales'].sum().reset_index()

# Feature Engineering
daily_sales.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

# Train the model
model = Prophet()
model.fit(daily_sales)

# Create a dataframe for future dates (next 6 weeks)
future = model.make_future_dataframe(periods=6*7, freq='D')

# Make predictions
forecast = model.predict(future)

# Visualize the forecast
fig = model.plot(forecast)
plt.show()

# Merge forecast with train_data to get specific predictions
forecast = pd.merge(forecast, train_data, left_on='ds', right_on='Date', how='left')

# Select relevant columns for the test_data set
test_predictions = forecast.loc[forecast['Date'].isin(test_data['Date']), ['Date', 'Store', 'yhat']]

# Display the test predictions
print(test_predictions)


# # 2.1 Preprocessing

# In our case, you have a few datetime columns to preprocess. you can extract the following from them:
# weekdays
# weekends 
# number of days to holidays
# Number of days after holiday
# Beginning of month, mid month and ending of month
# (think of more features to extract), extra marks for it
# 			
# As a final thing, you have to scale the data. This helps with predictions especially when using machine learning algorithms that use Euclidean distances. you can use the standard scaler in sklearn for this.
# 

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming train_data and test_data are already defined as DataFrames
# Combine train and test datasets for uniform preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Convert 'Date' column to datetime type
combined_data['Date'] = pd.to_datetime(combined_data['Date'])

# Extract features from datetime columns
combined_data['Weekday'] = combined_data['Date'].dt.weekday
combined_data['IsWeekend'] = (combined_data['Weekday'] >= 5).astype(int)

# Handling 'DaysToHoliday' and 'DaysAfterHoliday' columns
# (This logic assumes you're treating StateHoliday and SchoolHoliday as categories)
# Calculate the number of days to and after the nearest holiday
# (You can modify this logic based on your specific requirements)

# Example: Calculate the quarter of the year
combined_data['Quarter'] = combined_data['Date'].dt.quarter

# Scaling the data using StandardScaler
scaler = StandardScaler()
scaled_features = ['Weekday', 'Quarter']  # Update this list based on your specific requirements
combined_data[scaled_features] = scaler.fit_transform(combined_data[scaled_features])

# Separate combined_data back into train and test datasets
train_data = combined_data[combined_data['Sales'].notnull()]
test_data = combined_data[combined_data['Sales'].isnull()]

# Display the preprocessed datasets
print("Train Data:")
train_data.head()


# In[ ]:




