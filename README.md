# Unlocking_Youtube_Channel_Performance_Secrets_p2
## Project Overview
#### Project Tittle : Unlocking Youtunbe Channel Performance Secrets
#### Level : Intermediate
#### Tools : Visual Studio, Jupyter Notebook, Excel.
#### Languages : Python
#### Databases : `youtube_channel_real_performance_analytics`
This project focuses on unlocking the secrets behind YouTube channel performance by analyzing data using Python in Jupyter Notebook. The goal is to understand key engagement metrics such as views, likes, comments, watch time, and subscriber growth to identify trends that contribute to a channel's success. Using Python libraries like pandas, NumPy, matplotlib, and seaborn, the project involves cleaning, processing, and visualizing data to extract valuable insights.Ultimately, these insights empower content creators to refine their strategies and improve overall channel performance.

## Objectives
1. Set up a Youtube channel real performance database: Create and Analyze Unlocking Youtube Channel Performance Secrets database with the provided data.
2. Data Cleaning: Identify and remove any records with missing or null values.
3. Exploratory Data Analysis (EDA): Perform basic exploratory data analysis to understand the dataset.
4. Machine Learning:  Once the data is prepared, predictive modeling is implemented using machine learning techniques to uncover patterns, trends, and
actionable insights.

## 1. Python Project Structure
## About Dataset
This dataset provides an in-depth look at YouTube video analytics, capturing keymetrics related to video performance, audience engagement, revenue generation, and viewer behavior. Sourced from real video data, it highlights how variables like video duration, upload time, and ad impressions contribute to monetization and audience retention. This dataset is ideal for data analysts, content creators, and marketers aiming to uncover trends in viewer engagement, optimize content strategies, and maximize ad revenue. Inspired by the evolving landscape of digital content, it serves as a resource for understanding the impact of YouTube metrics on channel growth and content reach.

Video Details: Columns like Video Duration, Video Publish Time, Days Since Publish,
Day of Week.

#### Revenue Metrics: Includes Revenue per 1000 Views (USD), Estimated Revenue (USD), Ad Impressions, and various ad revenue sources (e.g., AdSense, DoubleClick).
#### Engagement Metrics: Metrics such as Views, Likes, Dislikes, Shares, Comments, Average View Duration, Average View Percentage (%), and Video Thumbnail CTR (%).
#### Audience Data: Data on New Subscribers, Unsubscribes, Unique Viewers, Returning Viewers, and New Viewers.
#### Monetization & Transaction Metrics: Details on Monetized Playbacks, Playback-Based CPM, YouTube Premium Revenue, and transactions like Orders and Total Sales Volume (USD).

### 1. Import Modules Libraries
```python
import numpy as np  #Manipulating Numbers
import pandas as pd #Dataframes
import seaborn as sns   #Visualisation
import matplotlib.pyplot as plt #Visualisation
%matplotlib inline 
sns.set(color_codes=True)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```
### 2. Load And Explore The Data
```python
# Load the data
data = pd.read_csv(r"C:\Users\thamm\OneDrive\Documents\Internship\Projects\Unlocking You Tube Channel Performance Secrets\youtube_channel_real_performance_analytics.csv")

# #Analyze the data
df = pd.DataFrame(data)
df

#Dimension of data
df.shape

#Basic information about the data
df.info()

#Rows
df.index

#Columns Names
df.columns

#Head of the data
df.head()

#Tail of the data
df.tail()

#Check for datatypes
df.dtypes
```

### 3. Data Cleaning and Preprocessing
#### Handling Missing Values
```python
#Check for missing values for each column
data.isnull().sum()

#Remove duplicates rows
data.drop_duplicates()

# Fill or drop null values
data.dropna()
```

#### Convert to Date Time Format
```python
# Convert video Publish Time 
df['Video Publish Time'] = pd.to_datetime(df['Video Publish Time'])
df['Video Publish Time']
```

### 4. Exploratory Data Analysis (EDA)
#### Bar Graph of Video Duration
```python
# Distribution of Video Duration
plt.figure(figsize=(8,5))
sns.histplot(data['Video Duration'], bins=30, color= 'blue', edgecolor='white', kde=True)
plt.title("Distribution of Video Duration", fontsize=14)
plt.xlabel=("Video Duration", "fontsize=12")
plt.ylabel("Frequency", fontsize=12)
plt.show()
```

#### Analysis Relationship
```python
# Pairplot to visualize relationships
sns.pairplot(data[['Revenue per 1000 Views (USD)', 'Views', 'Subscribers', 'Estimated Revenue (USD)']])
plt.show()
```

#### Correlation Analysis
```python
# Select only numeric columns
numeric_df = data.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr = numeric_df.corr()
```
```python
# plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap", fontsize=14)
plt.show()
```

#### Top Performers by Revenue
```python
top_videos = data.sort_values(by='Estimated Revenue (USD)', ascending=False).head(10)
print(top_videos[['ID', 'Estimated Revenue (USD)', 'Views', 'Subscribers']])
```

### Features
```python
# Create revenue per view
data['Revenue per View'] = data['Estimated Revenue (USD)']/data['Views']
data['Revenue per View']
```
```python
# Create engagement rate
data['Engagement Rate'] = (data['Likes'] + data['Shares'] + data['New Comments']) / data['Views'] * 100
data['Engagement Rate']
```

### 5. Data Visualization
```python
#Revenue Distribution:
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Revenue (USD)'], bins=50, kde=True, color='green')
plt.title("Revenue Distribution", fontsize=14)
plt.xlabel=("Revenue (USD)", 'fontsize=12')
plt.ylabel("Frequency", fontsize=12)
plt.show()
```
```python
# Revenue Vs Views
plt.figure(figsize=(10,6))
sns.scatterplot(x=data['Views'], y=data['Estimated AdSense Revenue (USD)'], alpha=0.7)
plt.title('Revenue Vs Views', fontsize=14)
plt.xlabel=('Views','fontsize=12')
plt.ylabel('Revenue (USD)', fontsize=12)
plt.show()
```

### 6. Predictive Model: Estimate Revenue
#### Prepare Data
```python
# Select features and target
features = ['Views', 'Subscribers', 'Likes', 'Shares', 'New Comments', 'Engagement Rate']
target = 'Estimated Revenue (USD)'

X = data[features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```python
X_train.head()
```
```python
X_test.head()
```
```python
y_train.head()
```
```pythom
y_test.head()
```

#### Train Random Forest Regressior
```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
y_pred
```
#### Evaluate the model
```python
# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
```

### 7. Insight and Recommendations
#### Use Visualization and feature importance to derive insights
```python
# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance", fontsize=14)
plt.show()
```
### 8. Discussion
In this notebook, we explored a comprehensive YouTube channel performance dataset. We visualized key metrics, examined correlations, and built a predictive model for estimating revenue. The Random Forest model provided a reasonable prediction accuracy, but there's always room for improvement. Future analysis could explore feature engineering, hyperparameter tuning, or even different modeling approaches to enhance prediction performance. If you found this analysis insightful, please consider upvoting this notebook.
