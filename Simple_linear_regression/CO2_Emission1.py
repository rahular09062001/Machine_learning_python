!wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("FuelConsumption.csv")
tdt=df[["ENGINESIZE","CO2EMISSIONS"]]
msk=np.random.rand(len(tdt)) < 0.7

train = tdt[msk]
test  = tdt[~msk]

#plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color = 'green')
#plt.show()

from sklearn import linear_model

rgr = linear_model.LinearRegression()
train_x=np.asanyarray(train[["ENGINESIZE"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])
rgr.fit(train_x,train_y)

print("Coefficent : ",rgr.coef_)
print("Intercept : ",rgr.intercept_)

plt.scatter(train_x,train_y,color="green")
plt.plot(train_x,train_x*rgr.coef_ + rgr.intercept_,color='blue')
plt.show()

from sklearn.metrics import r2_score

test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])
test_y_=rgr.predict(test_x)

print("Mean Absolute Error : ",np.mean(np.absolute(test_y_ - test_y)))
print("Residual Sum of Squares : ",np.mean((test_y_ - test_y)**2))
print("R2 Score : ",r2_score(test_y,test_y_))

plt.scatter(test_x,np.absolute(test_y_-test_y),color="orange")
plt.xlabel("Engine Size")
plt.ylabel("Absolute error")
plt.show()

plt.scatter(test_x,test_y,color="pink")
plt.plot(test_x,test_y_,color="red")
plt.xlabel("Engine Size")
plt.ylabel("Co2 Emission")
plt.show()
tout=pd.DataFrame({"EngineSize":test_x[0:,0],"CO2Emission":test_y[0:,0],"CO2Emission_Predicted":test_y_[0:,0]})
print(tout)
tout.to_csv("train_test_out2.csv")
print(len(tout))
df.hist()
plt.show()
