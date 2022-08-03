import pickle

file = open("Fuel-consumption-predictor.py", "r")

file_name='fuel_model.pkl'
f = open(file_name,'wb')
for x in file:
 pickle.dump(x,f)
f.close()
file.close()
