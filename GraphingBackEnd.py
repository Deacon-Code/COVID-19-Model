import pandas as pd

NYAllData = pd.read_csv(r'C:\Users\Tyler\Documents\Programming\Python Programs\DeaconCode\COVID-19 Modeling\COVID-19-Model\New_York_State_Statewide_COVID-19_Testing_20240306.csv')
NYAllData.loc[NYAllData['Geography Description'] == 'St. Lawrence', ['Geography Description']] = 'Saint Lawrence'

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime

counties=[]
for i in range(0, 72):
    if NYAllData.at[i, 'Geography Level'] == 'COUNTY':
        counties.append(NYAllData.at[i, 'Geography Description'])
    
NYAllData['Test Date +'] = pd.to_datetime(NYAllData['Test Date'])

for county in counties:
    SpecificCountyData = NYAllData.loc[NYAllData['Geography Description'] == county] 
    #Step 4: Building a model
    
    #doing something I read online to make this date time data actually usable
    #What is basically has done is make Test Date now be the number of days since the start going from 0 (days 1) to the end (the final recored day)
    #Very cheeky
    SpecificCountyData['Test Date +'] = np.arange(len(SpecificCountyData))


    #Adding a cosine feature
    SpecificCountyData['Cos'] = np.cos(2 * np.pi* SpecificCountyData['Test Date +'] / 365)

    X= SpecificCountyData[['Test Date +','Cos']]
    y = SpecificCountyData['Total Cases Per 100k']

    model = LinearRegression().fit(X,y)

    #adding the linear regression data for a county to the NYAllData dataFrame
    #Here is what happens, first a conditional runs to see if the matching county is found
    #then, once this occurs, the model adds a column called 'Linear Regression' to the NYAllData dataFrame
    #This column is then filled with the predicted values of the model
    NYAllData.loc[NYAllData['Geography Description'] == county, 'Linear Regression'] = model.predict(X) #CO-Pilot helped me with this line of code, I have no clue what is going on


    metrics.mean_absolute_error(model.predict(X), SpecificCountyData['Total Cases Per 100k'])
    print("MAE is: ", metrics.mean_absolute_error(model.predict(X), SpecificCountyData['Total Cases Per 100k']))


    x_axis= pd.to_datetime(SpecificCountyData['Test Date']) #This needed to be converted to a date time format to work, used CO-Pilot to help me with this
    y = SpecificCountyData['Total Cases Per 100k']


    plt.xlim(datetime(2020,3,1), datetime(2021,3,3))
    plt.ylim(0, 660)
    plt.scatter(x_axis, y, color = 'green')

    plt.plot(x_axis, model.predict(X), color="blue") #CO-Pilot added the model.predict(X) part, and I think I understand what it is doing
    plt.xlabel("Test Date")
    plt.ylabel("Total Cases Per 100k")
    plt.title(county +", MAE: "+ str(metrics.mean_absolute_error(model.predict(X), SpecificCountyData['Total Cases Per 100k'])))
    plt.show()
