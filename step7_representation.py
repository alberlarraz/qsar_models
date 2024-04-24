import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from random import random, sample, choice, uniform
from math import floor
from tqdm import tqdm
from numpy import array, dot, mean
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sys import exit

#SELECTION OF DESCRIPTORS FOR QSAR MODEL
descriptors = ['GATS3e',
 'ALogP',
  'Dv',
   #'BCUT.4',
    'GATS1c',
     'AVP-1',
      'RNCS33',
       #'CPSA.21',
        #'BCUT.2'
]

#INPUT DATA
print("Uploading data...")
data_training = pd.read_excel("Alergenos_training_set_1_OUT.xlsx")

#DATA NORMALIZATION
print("Normalizing data...")
mean_dict = {}
std_dict = {}
for desc in descriptors:
    desc_mean = np.mean(data_training[desc])
    desc_std = np.std(data_training[desc])
    z_score = [(x-desc_mean)/desc_std for x in data_training[desc]]
    mean_dict[desc] = desc_mean
    std_dict[desc] = desc_std
    data_training[desc] = z_score

mean_zscore = pd.DataFrame(mean_dict, index=[0])
std_zscore = pd.DataFrame(std_dict, index=[0])

#MULTIPLE LINEAR REGRESSION
inputs = (data_training[list(descriptors)].astype(float)).values.tolist()
outputs = data_training["log(EC3)"].values.tolist()
print("Fitting the model...")
model = LinearRegression().fit(inputs,outputs)
n = len(outputs)
individual_size = len(descriptors)
r_sq = model.score(inputs,outputs)
ad_r_sq = 1-((1-r_sq)*(n-1)/(n-individual_size-1))
intercept = model.intercept_
coefficients = model.coef_
cv_error = -cross_val_score(model, inputs, outputs, scoring='neg_mean_squared_error').mean()
predictions = model.predict(inputs)

#PLOT
print("Plotting...")
compound_names = data_training["COMP"]

plt.figure(figsize=(10,10))
plt.scatter(outputs, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

for i, name in enumerate(compound_names):
    plt.text(outputs[i], predictions[i], name)


diff = abs(outputs-predictions)
df = pd.DataFrame({'Compound Name': compound_names, 'Difference': diff, 'Actual Values':outputs , 'Predicted Values':predictions })
df.to_excel('diferencias.xlsx', index=False)

print("__FINAL MODEL__")
print("log(EC3): ", intercept,
 "+", coefficients[0], descriptors[0],
 "+", coefficients[1], descriptors[1],
  "+", coefficients[2], descriptors[2],
 "+", coefficients[3], descriptors[3],
  "+", coefficients[4], descriptors[4],
   "+", coefficients[5], descriptors[5],
#    "+", coefficients[6], descriptors[6],
#  "+", coefficients[7], descriptors[7],
#   "+", coefficients[8], descriptors[8]
)

print("STATISTICAL PARAMETERS")
print({"R2": r_sq, "Cross_Val": cv_error, "Predictibilidad": ad_r_sq})

print(mean_zscore)
print(std_zscore)

plt.show()