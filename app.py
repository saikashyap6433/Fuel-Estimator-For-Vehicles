import pickle
import jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split

data = pd.read_csv("measurements.csv")
data.head()
data.info()

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

# dropping Null values parameters
dropped_data = data.drop(['refill liters','refill gas','specials'],axis = 1)
dropped_data.info()
dropped_data.head()
delete_comma_and_convert_float(dropped_data, 'distance')
dropped_data['distance'] = dropped_data['distance'].astype(float)
delete_comma_and_convert_float(dropped_data, 'consume')
dropped_data['consume'] = dropped_data['consume'].astype(float)

# change 'gas_type' values to 1 and 0. String to int
dropped_data['gas_type'] = dropped_data['gas_type'].map({'SP98': 1, 'E10': 0})  


#### Getting the data ready

new_df = dropped_data[['distance','speed','temp_outside','gas_type','rain','sun','consume']]
sorted_df = new_df = new_df.sort_values('consume')
dataset_x = sorted_df.drop(['consume'],axis=1)
dataset_y = sorted_df.consume.values

x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.2, random_state= 42)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

reg=LinearRegression()

reg.fit(x_train,y_train)

filename = 'fuel_model.pkl'
pickle.dump(reg, open(filename, 'wb'))

app = Flask(__name__)
model = pickle.load(open('fuel_model.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

   if request.method == 'POST':
      distance=float(request.form['Distance'])
      speed=int(request.form['Speed'])
      #temp_inside=float(request.form['temp_inside'])
      temp_outside=float(request.form['temp_outside'])
      gas_type=request.form['Fuel_Type']
      Other_Factors=request.form['Other_Factors']
      if(Other_Factors=='AC'):
        Other_Factors_AC=1
        Other_Factors_sun=0
        Other_Factors_rain=0
      if(Other_Factors=='sun'):
        Other_Factors_AC=0
        Other_Factors_sun=1
        Other_Factors_rain=0
      if(Other_Factors=='rain'):
        Other_Factors_AC=0
        Other_Factors_sun=0
        Other_Factors_rain=1
      def lr(distance,speed,temp_outside,Other_Factors_AC,Other_Factors_sun,Other_Factors_rain):
       c=pd.DataFrame([distance,speed,temp_outside,Other_Factors_AC,Other_Factors_sun,Other_Factors_rain]).T
       return model.predict(c)
   prediction=lr(distance,speed,temp_outside,Other_Factors_AC,Other_Factors_sun,Other_Factors_rain)
   return render_template('index.html',prediction_text="Fuel estimate for car is {} liters".format(np.round(prediction,2)))

if __name__=="__main__":
    app.run(debug=True)
