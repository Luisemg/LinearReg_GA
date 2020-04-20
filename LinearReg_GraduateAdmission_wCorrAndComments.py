import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__errors__=[] # Variable to Store Errors if wanted to Plot them
__testedErrors__ =[]

def sustituir(w, b, x, index):
    """
        Function used to Substitute Values in a normal linear Function:
        such as: x1w1 + x2w2 + ... + xnwn = y
    """
    sub = 0
    for m in range(len(x)):
        sub = sub + (w[m]*x[m][index])
    sub = sub + b
    return sub

def sustituirMatrix(w, b, x):
    """
        Used to subtitute as Well, but instead of returning a single Value
        Returns a Matrix of hyphotetical y's
    """
    sub = 0
    for i in range(len(x)):
        sub = sub + (w[i]*x[i])
        sub = sub + b
    return sub

def cal_error(y, y_):
    """
        Function Used to Calculate the Error, uses Numerical Python to make
        A sum of Matrixes:
        Matrix y is the real result of the Data
        Matrix y_ is the hyphotetical y calculated with our weights
        This Function also stores the calculated error on a list that
        is used later to graph the MSE
    """
    N = y.shape[0]
    error = np.sum((y-y_)**2)/(N*2)
    __errors__.append(error)
    return error

def GD(w, b, alpha, x, y):
    """
        Gradient Descent Function:
        Used to calculate the rate of change on the weights making use of
        the error, and help aproximate the weights values to reach solution

        We use a temporal variable to store the actual weights and the new ones
        later, a temporal variable for the same purpose but with the bias.
        a variable to store the acumulated error, and a sum to indicate
        the summatory of the formula.

        y.shape[0] returns the length of the Matrix y
        param is used as a columns and dato is used as rows
        both temp (updated weights) and tempBias (updated Bias), are returned.
    """
    temp = list(w)
    tempBias = b
    for param in range(len(x)):
        sum = 0
        errorAcum = 0
        for dato in range(y.shape[0]):
            errorAcum = (sustituir(w, b, x, dato) - y[dato])
            sum = sum + (errorAcum * x[param][dato])
        temp[param] = w[param] - (alpha*(2.0/(y.shape[0]))*sum)
    sum = 0
    errorAcum = 0
    for dato in range(y.shape[0]):
        errorAcum = (sustituir(w, b, x, dato) - y[dato])
        sum = sum + errorAcum
    tempBias = b - (alpha*(2.0/y.shape[0])*sum)
    return temp, tempBias

"""
    The program Starts Here! :
"""
"""
    Variables used to take the Data from the xlsx File (Excel)
    datos = Data from the File to Train
    testing = Data from the File to Test

    I picked around 80% of the data to train, and 20% of it to Test
"""
datos = pd.read_excel('GA.xlsx', 'Training')
testing = pd.read_excel('GA.xlsx', 'Testing')
# print(datos)

# datos.plot.scatter(x='X1 N', y='Y N')
# plt.show()

"""
    Assignation of Variables to the x, and y, to make the matrixes of
    our data.
"""
# Data:
x1 = datos['X1 N'].values
x2 = datos['X2 N'].values
x3 = datos['X3 N'].values
x7 = datos['X7 N'].values
y = datos['Y N'].values

# Data For Testing
x1T = testing['X1 N'].values
x2T = testing['X2 N'].values
x3T = testing['X3 N'].values
x4T = testing['X7 N'].values
yT = testing['Y N'].values

# Defining the Parameters (weights) randomly
b = np.random.normal()
w1 = np.random.normal()
w2 = np.random.normal()
w3 = np.random.normal()
w7 = np.random.normal()

# Creating the list with the parameters and the samples of the Data
x = [x1, x2, x3, x7]
w = [w1, w2, w3, w7]
xT = [x1T, x2T, x3T, x4T]

# Defining the learning Rate and the number of Epochs
alpha = 0.01
epochs = 1500

# Defining the Error Vector
error = np.zeros(5)
print("Initial Weights: ")
print(w)
print("Initial Bias: ")
print(b)

# Training
for i in range(epochs):
    [w, b] = GD(w,b,alpha,x,y)
    y_ = sustituirMatrix(w,b,x)
    error = cal_error(y, y_)
    if (((i+1)%200 == 0) or i == 0 or i == 1 or i == 2):
        print("Parameters by Now on Epoch %i:" % (i+1))
        print(w)
        print("Error: ")
        print(error)
        print("=======================")

# Testing
averageError = 0; errorSum = 0
testErrors = []
yValues = np.zeros(len(yT))
yValuesReal = np.zeros(len(yT))
for t in range(len(yT)):
    testValue = sustituir(w, b, xT, t)
    valorPReal = (testValue * 22.6991286) + 66.6137845
    valorReal = (yT[t] * 22.6991286) + 66.6137845
    print("The expected value was: ", valorReal)
    print(" And we Got, ", valorPReal)
    averageError = abs(valorPReal - valorReal)
    errorSum = errorSum + averageError
    testErrors.append(averageError)
    yValues[t] = valorPReal
    yValuesReal[t] = valorReal

# Plotting the Epocs against de MSE
plt.subplot(1,4,1)
plt.plot(range(epochs), __errors__)
plt.title('Training Error (MSE)')
plt.xlabel('Training Epoch')
plt.ylabel('MSE')

# Plotting the error between Real Data and Tested Data
plt.subplot(1,4,2)
plt.plot(range(len(yT)), testErrors)
plt.title('Difference Between Tested and Real')
plt.xlabel('Tested X')
plt.ylabel('Error')

plt.subplot(1,4,3)
plt.plot(x[0], y, 'o')
plt.title('Original Training Data')
plt.xlabel('X values')
plt.ylabel('Y Values')

plt.subplot(1,4,4)
plt.plot(xT[0], yValues, 'o')
plt.title('Calculated Data')
plt.xlabel('Tested X')
plt.ylabel('Error')
plt.show()

print(errorSum/(2*len(yT)))
