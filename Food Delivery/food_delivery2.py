# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 08:47:07 2020

@author: HONEY
"""

import pandas as pd
import numpy as np


z_train = pd.read_excel("Data_Train.xlsx")
z_test = pd.read_excel("Data_Test.xlsx")

########Training the model#####################

#Clearing the unnecessary information
z_train.Rating = z_train.Rating.replace(r'[-a-zA-z_train ]+', np.nan, regex=True)
z_train.Votes = z_train.Votes.replace(r'[-a-zA-z_train ]+', np.nan, regex=True)
z_train.Reviews = z_train.Reviews.replace(r'[-a-zA-z_train ]+', np.nan, regex=True)
z_train.Minimum_Order = z_train.Minimum_Order.str.replace("₹","")
z_train.Average_Cost = z_train.Average_Cost.str.replace("₹","")
z_train.Average_Cost = z_train.Average_Cost.str.replace(",","")
z_train.Average_Cost = z_train.Average_Cost.replace(r'[-a-zA-z_train ]+', np.nan, regex=True)
z_train.Delivery_Time = z_train.Delivery_Time.str.replace(" minutes","")

# Filling the missing values
from sklearn.impute import SimpleImputer
missingValues = SimpleImputer(missing_values = np.nan, strategy= 'mean', verbose= 0)
missingValues = missingValues.fit(z_train[["Rating","Votes","Reviews","Average_Cost"]])
z_train[["Rating","Votes","Reviews","Average_Cost"]] = missingValues.transform(z_train[["Rating","Votes","Reviews","Average_Cost"]])

z_train =pd.concat([z_train, z_train.Cuisines.str.get_dummies(sep=", ")], 1)
z_train =pd.concat([z_train, z_train.Location.str.get_dummies(sep="\n")], 1)
Y = z_train.Delivery_Time
z_train = z_train.drop(["Wraps", "Cuisines", "Location","Delivery_Time","Restaurant"], axis=1)


X_train = z_train.iloc[:,:].values
Y_train = Y.values


from sklearn.ensemble import RandomForestClassifier
cl2 = RandomForestClassifier(n_estimators =20, random_state =0)
cl2.fit(X_train, Y_train)
y3 = cl2.predict(X_train)


#######################Testing out model###############################################

#Clearing the unnecessary information
z_test.Rating = z_test.Rating.replace(r'[-a-zA-z_test ]+', np.nan, regex=True)
z_test.Votes = z_test.Votes.replace(r'[-a-zA-z_test ]+', np.nan, regex=True)
z_test.Reviews = z_test.Reviews.replace(r'[-a-zA-z_test ]+', np.nan, regex=True)
z_test.Minimum_Order = z_test.Minimum_Order.str.replace("₹","")
z_test.Average_Cost = z_test.Average_Cost.str.replace("₹","")
z_test.Average_Cost = z_test.Average_Cost.str.replace(",","")
z_test.Average_Cost = z_test.Average_Cost.replace(r'[-a-zA-z_test ]+', np.nan, regex=True)


# Filling the missing values
z_test[["Rating","Votes","Reviews","Average_Cost"]] = missingValues.transform(z_test[["Rating","Votes","Reviews","Average_Cost"]])

z_test =pd.concat([z_test, z_test.Cuisines.str.get_dummies(sep=", ")], 1)
z_test =pd.concat([z_test, z_test.Location.str.get_dummies(sep="\n")], 1)
z_test = z_test.drop(["Wraps", "Cuisines", "Location","Restaurant"], axis=1)

common_columns = z_train.columns.union(z_test.columns)
z_test = z_test.reindex(columns = common_columns, fill_value = '0')


X_test = z_test.iloc[:,:].values

Y_pred = cl2.predict(X_test)

Y_pred = [ str(x)+" minutes" for x in Y_pred]

result = pd.DataFrame(Y_pred, columns=["Delivery_Time"])

result.to_excel("submission.xlsx", index = False)




