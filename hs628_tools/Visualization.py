# -*- coding: utf-8 -*-
"""
This is a module that outputs the corresponding visualization tool requested by the user. The module requires that the user has the train and test subsets already locally stored within the file (see bottom of file for details). Acquiring a plotly account is reccomended. 

"""
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
###This is importing a class from Issue #11 Test module. We used the ratio function to determine the test size ratio for the train_test_split function. 
#from Test import data_split as ds



def count_and_print(variable_name, x_label, title):
    """   
    This function counts all instances in a variable and creates a barplot to 
    display the results.
    
    Args: 
        variable_name: The name of the variable that is displayed
        x_label: The name of the variable displayed on the chart
        title: The title of the chart 
    
    Returns: A count bar plot of every instance of that variable 
    
    """
    
    def count(x1):
        """
        This function counts every instance of the variable and stores it in a dictionary
        
        Args: 
            x1: The variable to be counted 
        Returns: 
            A dictionary of the variable. The key is the instance to be counted and the value
            is the count value of that instance 
        """
        new_dic = {}
        c = (len(x1)-1)
        # looping through the list
        for index, value in enumerate(x1):
            count = 0
            # checks to see if the value was in list, if it is then goes to the next index
            if((value in new_dic) == True):
                pass
            else:
                # accumulates the count for that unique value 
                for index2, value2 in enumerate(x1):
                    if value == value2:
                        count = count + 1
                    if index2 == (len(x1)-1):
                        new_dic[value] =  count
        return new_dic
    
    count_dictionary = count(variable_name)
    from collections import OrderedDict
    from operator import itemgetter

    count_ordered_list = (OrderedDict(sorted(count_dictionary.items(), key = itemgetter(0), reverse = False)))

    def key_list(ordered_dictionary):
        """
        This function takes in a dictionary and returns a list of the keys only 
        
        Args:
            ordered_dictionary: An ordered dictionary of the keys and values needed
            to be extracted
        Returns: 
            keys_list: A list of the keys within that dictionary 
        """
        keys_list= []
        for key in ordered_dictionary.keys():
            keys_list.append(key)
        return keys_list

    def value_list(ordered_dictionary):
        """
        This function takes in a dictionary and returns a list of the values only 
        
        Args:
            ordered_dictionary: An ordered dictionary of the keys and values needed
            to be extracted
        Returns: 
            values_list: A list of the values within that dictionary 
        """
        values_list= []
        for value in ordered_dictionary.values():
            values_list.append(value)
        return values_list
    
    keys = key_list(count_ordered_list)
    values = value_list(count_ordered_list)
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


def scatter(x,y,colorby,xlabel,ylabel,title,new_data):
    """
    This function outputs a scatter plot with respects to user input. The user can specify
    if they want to color each data point by a categorical variable.
    
    Args: 
        x: The x variable
        y: The y variable
        colorby: categorical variable in which the data points will be displayed by color
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        new_data: The data set being used 
    
    Returns: 
        A scatter plot with respective labels.
    
    """
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

def boxplot(x,y, x_label, y_label, title):
    """
    This function creates a box plot of the two variables the user decides to examine
    Args: 
        x: The x variable
        y: The y variable
        x_label: The label of the x-axis
        y_label: The label of the y-axis
        title: The title of the chart
    
    Returns: 
        A box plot with respective labels.
    
    """
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
        Returns: The metrics of the linear regression ran 
            
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
        
        Returns: A graphical representation of the linear model outputted
            
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


def question(data2,X_train,X_test,y_train,y_test, plot_type):
    """
    This function takes in user input and runs whatever plot or visualization tool entered
    
    Args: 
        data2: The data set being used
        X_train: x training set
        X_test: x testing set 
        y_train: y training set
        y_test: y testing set
        plot_type: The type of plot the user wishes to see prompted from question() function
    
    Returns: The plot requested by the user 
    """
    if plot_type == '2':
        x1 = input("Please enter X variable:")
        x = data2[x1]
        x2 = input("Please enter X2 variable (for comparison):")
        xx = data2[x2]
        y = input("Please enter Y variable:")
        x_lab = input("Please enter the label for the X-Axis:")
        y_lab = input("Please enter the label for the Y-Axis:")
        title = input("Please enter the title of the graph:")
        return scatter(x,y, xx, x_lab, y_lab, title,data2);

    if plot_type == '1':
        x1 = input("Please enter a variable:")
        #x = data2[x1]
        x_lab = input("Please enter the label for the X-Axis:")
        title = input("Please enter the title of your graph:")
        return count_and_print(data2[x1], x_lab, title);


    if plot_type == '3':
        x = data2[input("Please enter X variable name:")]
        y = data2[input("Please enter Y variable name:")]
        x_lab = input("Please enter the label for the X-Axis:")
        y_lab = input("Please enter the label for the Y-Axis:")
        title = input("Please enter the title of your graph:")
        return boxplot(x,y,x_lab,y_lab,title);
    
    if plot_type == '4':
        return model_metrics(X_train, X_test, y_train, y_test);
    
    if plot_type == '5':
        Q2= input("What do you want to title the graph?"+"\n")
        Q3 = input("What is the predictor(dependent) variable name?:"+"\n")
        Q4 = input("Name of x axis:")
        Q5 = input("Name of y axis:") 
        Q6 = input("Choose color for actual y points:"+"\n")
        Q7 = input("Choose color for line:"+"\n")
        model_metrics(X_train, X_test, y_train, y_test)
        return plot_model(X_train, X_test, y_train, y_test,Q2,Q3,Q4,Q5,Q6,Q7)

def prompt_user():
    """
    This function prompts the user for their information on the plot they wish to see
    and returns the respective plot 
    """
    
    Q1= input("What type of plot output? (Enter #)"+"\n"+"\n"+"1.Countplot"+"\n"+"2.Scatterplot"+"\n"+"3.Boxplot"+"\n"+"4.Simple Linear Regression Model"+"\n"+"5.Multiple Linear Regression Model"+"\n")
    question(data,Xtrain,Xtest,ytrain,ytest, Q1)

"""
Here we assume that the user already has these variables stored locally as np.arrays.
"""
#data = pd.read_csv('data.csv')
#sub_data = data.iloc[1:10,1:10]
#X = sub_data.iloc[:,0:3]
#Y = sub_data['readmission_30d']

#Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, test_size = .5) #This assumes that Issue #11 Test.py module works and outputs the test ratio. 
#Xtrain = np.array(Xtrain)
#ytrain = np.array(ytrain)
#Xtest = np.array(Xtest)
#ytest = np.array(ytest)
#prompt_user()


#count_and_print(data['readmission_30d'], "hey", "whats")





