# %%
"""
# Multiple Linear Regression Model
"""

# %%
"""
## Importing the required Libraries 
"""

# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# %%
"""
### Reading the data 
"""

# %%
data = pd.read_csv("measurements.csv")
data.head()

# %%
data.info()

# %%
"""
## Cleaning The data 
"""

# %%
"""
**Drop alert** - Columns containing all NULL values can be dropped as they have no effect on any other parameters
"""

# %%
# dropping Null values parameters
dropped_data = data.drop(['refill liters','refill gas','specials'],axis = 1)

# %%
dropped_data.info()

# %%
dropped_data.head()

# %%
"""
##### Looking at the data it has been observed that columns or parameters have values with commas in it. These are required to be replaced by dots.
"""

# %%
def delete_comma_and_convert_float(df,column_name):
    
    index = df.columns.get_loc(column_name)
    
    for i in range(len(df[column_name])):
        value = df.iloc[i,index]
        value_list = value.split(',')
        
        if len(value_list) == 2:
            new_value = float(''.join(value_list)) / 10
            df.iloc[i,index] = new_value
            
        else :
            df.iloc[i,index] = float (value)    

# %%
delete_comma_and_convert_float(dropped_data, 'distance')
dropped_data['distance'] = dropped_data['distance'].astype(float)
delete_comma_and_convert_float(dropped_data, 'consume')
dropped_data['consume'] = dropped_data['consume'].astype(float)

# %%
dropped_data['gas_type'] = dropped_data['gas_type'].map({'SP98': 1, 'E10': 0})  # change 'gas_type' values to 1 and 0. String to int

# %%
"""
#### Getting the data ready
"""

# %%
new_df = dropped_data[['distance','speed','temp_outside','gas_type','rain','sun','consume']]
sorted_df = new_df = new_df.sort_values('consume')
dataset_x = sorted_df.drop(['consume'],axis=1)
dataset_y = sorted_df.consume.values

# %%
"""
## Spotting any correlation between the parameters
"""

# %%
sns.heatmap(new_df.corr(),cmap = 'BrBG', annot=True)

# %%
plt.figure(figsize=(19,6))

plt.subplot(161)
plt.scatter(dataset_x['distance'],dataset_y)
plt.subplot(162)
plt.scatter(dataset_x['speed'],dataset_y)
plt.subplot(163)
plt.scatter(dataset_x['temp_outside'],dataset_y)
plt.subplot(164)
plt.scatter(dataset_x['gas_type'],dataset_y)
plt.subplot(165)
plt.scatter(dataset_x['rain'],dataset_y)
plt.subplot(166)
plt.scatter(dataset_x['sun'],dataset_y)


#plt.show()

# %%
"""
## Creating the data model
"""

# %%
"""
### Splitting the data
"""

# %%
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.2, random_state= 42)

# %%
x_train.shape

# %%
x_test.shape

# %%
y_train.shape

# %%
y_test.shape

# %%
"""
### Linear Regression Model
"""

# %%


# %%
Model = LinearRegression()

# %%
Model.fit(x_train,y_train)

# %%
#To retrieve the intercept:
print(Model.intercept_)

#For retrieving the slope:
print(Model.coef_)

# %%
"""
**Predicting Values for Train datset**
"""

# %%
y_pred = Model.predict(x_test)

# %%
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

# %%
"""
**The graph shows the actual values and the predicted values**
"""

# %%
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

# %%
"""
## Evaluation of the Model
"""

# %%


# %%
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# %%
"""
**Conclusion** : The RMSE value , MAE Value and the MSE value are ranging from 0.7 to 1.3 which is quite less. This means that the model is accurate enough to make good predictions
"""
