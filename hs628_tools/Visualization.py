import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys 
import plotly.plotly as py
import plotly.graph_objs as go
#plotly.tools.set_credentials_file(username='hhowell', api_key='BRP1aHhfTHwkqnvalVQL')
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
###This is importing a class from Issue #11 Test module. We used the ratio function to determine the test size ratio for the train_test_split function. 
#from Test import data_split as ds


###Bar Plot-Count
#This function counts all instances in a variable and creates a barplot to display the results 
#@param: The variable to be counted, the x_axis label, title of the graph
#@return: A count bar plot of the variable inputted 
def count_and_print(variable, x_label, title):
    def count(variable):
        new_dic = {}
        c = (len(variable)-1)
        for index, value in enumerate(variable):
            count = 0
            if((value in new_dic) == True):
                pass
            else:
                for index2, value2 in enumerate(variable):
                    if value == value2:
                        count = count + 1
                    if index2 == (len(variable)-1):
                        new_dic[value] =  count
        return new_dic

    variable = variable

    count_dictionary = count(variable)
    from collections import OrderedDict
    from operator import itemgetter

    count_ordered = (OrderedDict(sorted(count_dictionary.items(), key = itemgetter(0), reverse = False)))

    def key_list(list):
        keys_list= []
        for key in list.keys():
            keys_list.append(key)
        return keys_list

    def value_list(list):
        values_list= []
        for value in list.values():
            values_list.append(value)
        return values_list
    keys = key_list(count_ordered)
    values = value_list(count_ordered)
#Plot the variable
    f1 = (go.Bar(
                x=keys,
                y=values
        )
           )
    layout = go.Layout(
        title= title,
        xaxis=dict(
            title=x_label,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
                )
        ),
        yaxis=dict(
            title='Count',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
                )
            )
    )

    
    data = [f1]
    fig = go.Figure(data=data,layout=layout)
    return py.iplot(fig, filename='basic-bar')


###Scatter Plot 
#This function outputs a scatter plot with respects to user input. The user can specify
#if they want to color each data point by a categorical variable
#@param: x variable, y variable, x and y axis labels and the title of the plot
#@returns: a scatter plot
def scatter(x,y,colorby,xlabel,ylabel,title,new_data):
    trace1 = go.Scatter(
        x = x,
        y = y,
        mode='markers',
        marker=dict(
            size=16,
            color = new_data[colorby], #set color equal to a variable
            colorscale='Viridis',
            showscale=True
        )
    )
    layout = go.Layout(
        title= title,
        xaxis=dict(
            title=xlabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
                )
            ),
        yaxis=dict(
            title= ylabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
                )
            )
    )
    data = [trace1]
    fig = go.Figure(data=data,layout=layout)
    return py.iplot(fig, filename='scatter-plot-with-colorscale');

###BoxPlot
#This function creates a box plot of the two variables the user decides to examine
#@param: the x and y vales, along with the labels and the title of the graph
#@return: A box plot
def boxplot(x,y, y_label, x_label, title):
    data = [go.Box(x= x,
            y= y)]

    layout = go.Layout(
        title = title,
       yaxis=dict(
            title= y_label,
            zeroline=False
        ),
        xaxis=dict(
            title=x_label,
            zeroline=False
        )
    )

    fig = go.Figure(data=data,layout=layout)
    return py.iplot(fig, filename='test');


def model_metrics(X_train, X_test, y_train, y_test):
    """
    This function will print out the linear model metrics including the coefficients, mean squared error, mean absolute error, root mean squared error, and Variance score.

        Parameters:
            X_train: dataframe with training set split of all independent variables 
            y_train: dataframe with training set split of the dependent variable
            X_test: dataframe with the testing set split of all the independent variables
            y_test: dataframe with the testing set split of the dependent variable
            
    """
    reg_obj = linear_model.LinearRegression()
    reg_obj.fit(X_train, y_train)
    y_pred = reg_obj.predict(X_test)

    # The coefficients
    print('Coefficient(s): \n', reg_obj.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # The mean absolute error 
    print("Mean absolute error: %.2f"
          % mean_absolute_error(y_test, y_pred))
    # The mean squared error
    print("Root mean squared error: %.2f"
          % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))


def plot_model(X_train, X_test, y_train, y_test, Q2, Q3, Q4, Q5, Q6, Q7):
    """
    This function will plot out the simple linear regression model with the points as the actual testing set y values and the regression line to compare predicted y values. 

        Parameters:
            X_train: dataframe with training set split of all independent variables 
            y_train: dataframe with training set split of the dependent variable
            X_test: dataframe with the testing set split of all the independent variables
            y_test: dataframe with the testing set split of the dependent variable
            
    """
    reg_obj = linear_model.LinearRegression()
    reg_obj.fit(X_train, y_train)
    y_pred = reg_obj.predict(X_test)
    
    def data_to_plotly(x):
        k = []
    
        for i in range(0, len(x)):
            k.append(x[i][0])
        
        return k

    p1 = go.Scatter(x=data_to_plotly(X_test), 
                y=y_test, 
                mode='markers',
                marker=dict(color=Q6),
                name ='y_test values'
               )

    p2 = go.Scatter(x=data_to_plotly(X_test), 
                y=reg_obj.predict(X_test),
                mode='lines',
                line=dict(color=Q7, width=3),
                name ='Linear Regression Model'
                )

    layout = go.Layout(xaxis=dict(ticks='', showticklabels=True,
                              zeroline=False, title=Q4),
                   yaxis=dict(ticks='', showticklabels=True,
                              zeroline=False, title=Q5),
                   showlegend=True, hovermode='closest',
                   title=Q2
                   )

    fig = go.Figure(data=[p1, p2], layout=layout)

    return py.iplot(fig)


def question(data2,X_train,X_test,y_train,y_test, Q1):
    if Q1 == '2':
        x1 = input("Please enter X variable:")
        x = data2[x1]
        x2 = input("Please enter X2 variable (for comparison):")
        xx = data2[x2]
        y = input("Please enter Y variable:")
        x_lab = input("Please enter the label for the X-Axis:")
        y_lab = input("Please enter the label for the Y-Axis:")
        title = input("Please enter the title of the graph:")
        return scatter(x,xx, y, x_lab, y_lab, title,data2);

    if Q1 == '1':
        x1 = input("Please enter a variable:")
        x = data2[x1]
        x_lab = input("Please enter the label for the X-Axis:")
        title = input("Please enter the title of your graph:")
        return count_and_print(x, x_lab, title);


    if Q1== '3':
        x = data2[input("Please enter X variable name:")]
        y = data2[input("Please enter Y variable name:")]
        x_lab = input("Please enter the label for the X-Axis:")
        y_lab = input("Please enter the label for the Y-Axis:")
        title = input("Please enter the title of your graph:")
        return boxplot(x,y,x_lab,y_lab,title);
    
    if Q1 == '4':
        return model_metrics(X_train, X_test, y_train, y_test);
    
    if Q1 == '5':
        Q2= input("What do you want to title the graph?"+"\n")
        Q3 = input("What is the predictor(dependent) variable name?:"+"\n")
        Q4 = input("Name of x axis:")
        Q5 = input("Name of y axis:") 
        Q6 = input("Choose color for actual y points:"+"\n")
        Q7 = input("Choose color for line:"+"\n")
        model_metrics(X_train, X_test, y_train, y_test)
        return plot_model(X_train, X_test, y_train, y_test,Q2,Q3,Q4,Q5,Q6,Q7)

##### MAIN #####

##Note these four lines of code are meant to test this file 
#data = pd.read_csv('data.csv')
#sub_data = data.iloc[1:10,1:10]
#X = sub_data.iloc[:,0:3]
#Y = sub_data['readmission_30d']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, test_size = .5) #This assumes that Issue #11 Test.py module works and outputs the test ratio. 
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
Xtest = np.array(Xtest)
ytest = np.array(ytest)

Q1= input("What type of plot output? (Enter #)"+"\n"+"\n"+"1.Countplot"+"\n"+"2.Scatterplot"+"\n"+"3.Boxplot"+"\n"+"4.Simple Linear Regression Model"+"\n"+"5.Multiple Linear Regression Model"+"\n")
question(data,Xtrain,Xtest,ytrain,ytest, Q1)


 

