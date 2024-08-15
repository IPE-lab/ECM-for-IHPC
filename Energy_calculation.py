import xlwt
import xlrd
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.size'] = 9  # Display Chinese characters
mpl.rcParams['axes.unicode_minus'] = False
# Set the direction of xtick and ytick: in, out, inout
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


##========with curve_fit  normal working condition
def func(X, a,b,d,g):
    x1,x2,n,T = X
    return a*x1+b*x2+d*n+g*T

# ##========with curve_fit  air-to-air condition
def func(X, a, b):
    x1,x2 = X
    return a*x1+b*x2

## Read all sheets in Excel
path = 'E:\\yourfile.xlsx'
df = pd.read_excel(path, sheet_name = 'sheet_name')
print(df.shape)
print(df.head)
df.columns = [ 'y', 'x1', 'x11', 'x2', 'x22', 'x3', 'x33', 'n','t','T','n1','n2','n3']
x = sm.add_constant(df.iloc[:,1:]) # Generate independent variables
x1 = df['x11']
x2 = df['x22']
n = df['n']
x3 = df['x3']
n1 = df['n1']
n2 = df['n2']
n3 = df['n3']
y = df['y']
T = df['T']

#======with statsmodels
x = sm.add_constant(x) # adding a constant
model = sm.OLS(y, x) # Generating Independent Variables
result = model.fit() # Model fitting
a = result.summary()  # Model description
#b = result.summary2()
print(a)
#print(b)

#======with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

#========with curve_fit   bounds=([0,1,8], [1, np.inf, np.inf])
p0 = 7, 11
popt, pcov = curve_fit(func, (x1,x2,n1,n2,n3,T), y, bounds=([0,1.4099457,0,0,0,0], [0.00001,1.4099458,np.inf,np.inf,np.inf,np.inf]) )
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y, func((x1,x2,n1,n2,n3,T), *popt))
print(popt)
x = np.linspace(0,len(y),len(y))
plt.plot(x, y, 'b-', label='measured value')
plt.plot(x, func((x1,x2,n1,n2,n3,T), *popt), 'r-',label= r2, alpha=0.5)
plt.legend()
mse = mean_squared_error(y, func((x1,x2,n,x3), *popt))
rmse = np.sqrt(mse)
print(mse,rmse)
plt.show()



















