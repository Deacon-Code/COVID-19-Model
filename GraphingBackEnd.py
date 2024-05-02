#Some notes up front
#1) There are a whole bunch of imports here, so you might need to download the respective libraries for this to work
#2) This is a back end file, so it is not meant to be run on its own, but rather to be called by the front end, but you already knew that
#3) There is a specific data files being used. It is as follows
#       New_York_State_Statewide_COVID-19_Testing_20240306.csv
#4) You will need to get user input for 2 big things:
        # County
        # Date Range
     

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime

#this function will keep running until a county is found
#will search the lsit of counties
#If the county is found, it will return the county
def getCounty(county, counties):
    foundCounty = False
    while foundCounty == False:
        if county in counties:
            foundCounty = True
            return county
        else:
            print("County not found, please try again")
            county = input("Enter the county you would like to see: ")

#This function will keep running until the date range is in the correct format
#Makes recursive calls until the date is in the correct format
#Returns a tuple (aka a set of values) of the start and end date
def getDateRange(dateRangeStart, dateRangeEnd):
    try:
        startDate = pd.to_datetime(dateRangeStart)
        endDate = pd.to_datetime(dateRangeEnd)
        while(startDate > endDate or startDate == endDate or startDate < pd.to_datetime('2020-03-01') or endDate > pd.to_datetime('2024-03-03')):
            print("Start date is after end date, or the date is not in the correct range, please try again")
            print("The correct range is from 2020-03-01 to 2024-03-03")
            dateRangeStart = input("Enter the start date range. Must be of the format YYYY-DD-MM:")
            dateRangeEnd = input("Enter the end date range. Must be of the format YYYY-DD-MM:")
            startDate = pd.to_datetime(dateRangeStart)
            endDate = pd.to_datetime(dateRangeEnd)
        return startDate, endDate
    except:
        print("Date range not in correct format, please try again")
        dateRangeStart = input("Enter the start date range. Must be of the format YYYY-DD-MM:")
        dateRangeEnd = input("Enter the end date range. Must be of the format YYYY-DD-MM:")
        return getDateRange(dateRangeStart, dateRangeEnd)

#this opens the file and reads it into a pandas dataFrame
#You will probably want to change the file path if this does not work, but I tried to make it such that it is local
NYAllData = pd.read_csv(r'COVID-19-Model\New_York_State_Statewide_COVID-19_Testing_20240306.csv')

#This changes the naming of something in the dataFrame, from St. Lawrence to Saint Lawrence
NYAllData.loc[NYAllData['Geography Description'] == 'St. Lawrence', ['Geography Description']] = 'Saint Lawrence'


#This is a way to convert the date range into a format that the dataFrame can use
    
NYAllData['Test Date +'] = pd.to_datetime(NYAllData['Test Date'])


#Counties is a list of all of the counties. I do not know if this is useful, but it is here
#you might want to integrate this into a drop down menu or something
counties=[]
for i in range(0, 72):
    if NYAllData.at[i, 'Geography Level'] == 'COUNTY':
        counties.append(NYAllData.at[i, 'Geography Description'])

countySelected = getCounty(input("Enter the county you would like to see: "), counties) #This is the county that the user wants to see
startDate, endDate = getDateRange(input("Enter the start date range. Must be of the format YYYY-DD-MM:"), input("Enter the end date range. Must be of the format YYYY-DD-MM:")) #This is the date range that the user wants to see



#This sets the data in the data frame to the county selecteced, from that text input found above
SpecificCountyData = NYAllData.loc[NYAllData['Geography Description'] == countySelected] 
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
NYAllData.loc[NYAllData['Geography Description'] == countySelected, 'Linear Regression'] = model.predict(X) #CO-Pilot helped me with this line of code, I have no clue what is going on


metrics.mean_absolute_error(model.predict(X), SpecificCountyData['Total Cases Per 100k'])
print("MAE is: ", metrics.mean_absolute_error(model.predict(X), SpecificCountyData['Total Cases Per 100k']))


x_axis= pd.to_datetime(SpecificCountyData['Test Date']) #This needed to be converted to a date time format to work, used CO-Pilot to help me with this
y = SpecificCountyData['Total Cases Per 100k']


#SO RIGHT HERE IS VERY IMPORTANT
#This controls where the date start ends 
plt.xlim(startDate, endDate)
plt.ylim(0, 660)
plt.scatter(x_axis, y, color = 'green')

plt.plot(x_axis, model.predict(X), color="blue") #CO-Pilot added the model.predict(X) part, and I think I understand what it is doing
plt.xlabel("Test Date")
plt.ylabel("Total Cases Per 100k")
plt.title(countySelected +", MAE: "+ str(metrics.mean_absolute_error(model.predict(X), SpecificCountyData['Total Cases Per 100k'])))
plt.show()
