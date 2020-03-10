---
date: 2020-3-01
title: Assignment1
---


# Assignment1 —— Polynomial regression

```python
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import sklearn
import prettytable as pt
import numpy as np
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize,StandardScaler,PolynomialFeatures  
from sklearn import model_selection
from sklearn.linear_model import LinearRegression 
```


```python
res = {}
```

## a. Generate 20 data pairs (X, Y) using y = sin(2\*pi\*X) + N 


```python
x_sin = np.linspace(0,1,200)
x = np.linspace(0,1,20)
d = np.random.normal(loc=0,scale=0.2,size=20)    # N from the normal gaussian distribution 
print(d)
y_sin = 2*math.pi*x_sin
y = 2*math.pi*x
print(y)
for i in range(200):
    y_sin[i] = math.sin(y_sin[i])    
for i in range(20):
    y[i] = math.sin(y[i])+ d[i]
```

    [-0.10065512 -0.20123299  0.13067115 -0.35995294 -0.34811604  0.21112207
     -0.23787448  0.10021735 -0.01532149  0.01159478  0.01675671 -0.2049862
     -0.26199684 -0.08677789 -0.14949241  0.11808432  0.47300011 -0.10460325
      0.2606299   0.04766576]
    [0.         0.33069396 0.66138793 0.99208189 1.32277585 1.65346982
     1.98416378 2.31485774 2.64555171 2.97624567 3.30693964 3.6376336
     3.96832756 4.29902153 4.62971549 4.96040945 5.29110342 5.62179738
     5.95249134 6.28318531]



```python
plt.plot(x_sin, y_sin, "r-")  # original sin function curve
plt.scatter(x, y)
```




    <matplotlib.collections.PathCollection at 0x1a490bf490>




![png](./Hengchao_01_4_1.png)



```python
data_1 = {'X':x, 'Y':y}
data = pd.DataFrame(data = data_1, dtype=np.int8)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-0.100655</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.052632</td>
      <td>0.123466</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.105263</td>
      <td>0.744884</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.157895</td>
      <td>0.477214</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.210526</td>
      <td>0.621284</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.263158</td>
      <td>1.207707</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.315789</td>
      <td>0.677899</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.368421</td>
      <td>0.835941</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.421053</td>
      <td>0.460626</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.473684</td>
      <td>0.176189</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.526316</td>
      <td>-0.147838</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.578947</td>
      <td>-0.680934</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.631579</td>
      <td>-0.997721</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.684211</td>
      <td>-1.002551</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.736842</td>
      <td>-1.146077</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.789474</td>
      <td>-0.851316</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.842105</td>
      <td>-0.364166</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.894737</td>
      <td>-0.718816</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.947368</td>
      <td>-0.064070</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.000000</td>
      <td>0.047666</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train,X_test, Y_train, Y_test =model_selection.train_test_split(x, y, test_size=0.5, random_state=3)
```


```python
train = {'X':X_train, 'Y': Y_train}
train_data = pd.DataFrame(data = train, dtype=np.int8)
train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.473684</td>
      <td>0.237901</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.578947</td>
      <td>-0.421769</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>0.202538</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.947368</td>
      <td>-0.162660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.684211</td>
      <td>-0.950614</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.263158</td>
      <td>1.167077</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000000</td>
      <td>0.100110</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.421053</td>
      <td>0.415397</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.157895</td>
      <td>0.648296</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.526316</td>
      <td>-0.159142</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = {'X':X_test, 'Y': Y_test}
test_data = pd.DataFrame(data = test, dtype=np.int8)
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.736842</td>
      <td>-1.170818</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.105263</td>
      <td>0.519501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.052632</td>
      <td>0.682805</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.894737</td>
      <td>-0.622776</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.210526</td>
      <td>0.919025</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.842105</td>
      <td>-0.838238</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.315789</td>
      <td>1.068300</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.368421</td>
      <td>0.677289</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.789474</td>
      <td>-0.755381</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.631579</td>
      <td>-0.913760</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(X_train, Y_train)
plt.scatter(X_test, Y_test, c = 'r')
```




    <matplotlib.collections.PathCollection at 0x1a4b070f50>




![png](./Hengchao_01_9_1.png)


## b. Using room mean square error, find weights of polynomial regression for order is 0, 1, 3, 9


```python
def  polynomialRegression(i:int ) :
    polynomial = PolynomialFeatures(degree = i)# quadratic polynomial
    x_transformed = polynomial.fit_transform(X_train.reshape(10,1))
    poly_linear_model = LinearRegression()
    poly_linear_model.fit(x_transformed, Y_train)# train
    return polynomial, poly_linear_model
```

### weights of polynomial regression for order is 0


```python
polynomial_0, poly_linear_model_0 = polynomialRegression(0) 

coef = poly_linear_model_0.coef_
tmp = [0]*10
for i in range(len(coef)) :
    tmp[i] = int(coef[i])
res['0'] = tmp
coef
```




    array([0.])



### weights of polynomial regression for order is 1


```python

polynomial_1, poly_linear_model_1 = polynomialRegression(1)
 
coef = poly_linear_model_1.coef_
tmp = [0]*10
for i in range(len(coef)) :
    tmp[i] = int(coef[i])
res['1'] = tmp
coef
```




    array([ 0.        , -0.84393666])



### weights of polynomial regression for order is 3


```python
polynomial_3, poly_linear_model_3 = polynomialRegression(3)
 
coef = poly_linear_model_3.coef_
tmp = [0]*10
for i in range(len(coef)) :
    tmp[i] = int(coef[i])
res['3'] = tmp
coef
```




    array([  0.        ,  10.54357363, -32.04219135,  21.74789823])



### weights of polynomial regression for order is 9


```python
polynomial_9, poly_linear_model_9 = polynomialRegression(9)
coef = poly_linear_model_9.coef_
tmp = [0]*10
for i in range(len(coef)) :
    tmp[i] = int(coef[i])
res['9'] = tmp
coef
```




    array([ 0.00000000e+00, -9.15509033e+02,  1.78726994e+04, -1.41897131e+05,
            6.07692204e+05, -1.54842153e+06,  2.41692818e+06, -2.26586227e+06,
            1.17006720e+06, -2.55463736e+05])



## c. Display weights in table 


```python
from prettytable import PrettyTable
x= PrettyTable()
x.add_column("label\order", ["W0","W1","W2","W3","W4","W5","W6","W7","W8","W9"])
x.add_column("0", res["0"])
x.add_column("1", res["1"])
x.add_column("3", res["3"])
x.add_column("9", res["9"])
print(x)
# the label 0, W0 in the table is the weights of polynomial regression for order is 0
# the label 1, W0 and W1 in the table is the weights of polynomial regression for order is 1
# the label 3, W0, W1, W2 and W3 in the table is the weights of polynomial regression for order is 3
# the label 9, W0-W9 in the table is the weights of polynomial regression for order is 9
```

    +-------------+---+---+-----+----------+
    | label\order | 0 | 1 |  3  |    9     |
    +-------------+---+---+-----+----------+
    |      W0     | 0 | 0 |  0  |    0     |
    |      W1     | 0 | 0 |  10 |   -915   |
    |      W2     | 0 | 0 | -32 |  17872   |
    |      W3     | 0 | 0 |  21 | -141897  |
    |      W4     | 0 | 0 |  0  |  607692  |
    |      W5     | 0 | 0 |  0  | -1548421 |
    |      W6     | 0 | 0 |  0  | 2416928  |
    |      W7     | 0 | 0 |  0  | -2265862 |
    |      W8     | 0 | 0 |  0  | 1170067  |
    |      W9     | 0 | 0 |  0  | -255463  |
    +-------------+---+---+-----+----------+


## d. Draw a chart of fit data
### weights of polynomial regression for order is 0


```python
xx = np.linspace(0, 1, 100)
xx_transformed_0 = polynomial_0.fit_transform(xx.reshape(xx.shape[0], 1))
yy = poly_linear_model_0.predict(xx_transformed_0)
plt.plot(xx, yy,label="$y = N$")
plt.scatter(X_train, Y_train)
plt.scatter(X_test, Y_test, c = 'r')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a45e81a10>




![png](./Hengchao_01_23_1.png)


### weights of polynomial regression for order is 1


```python
xx = np.linspace(0, 1, 100)
xx_transformed_1 = polynomial_1.fit_transform(xx.reshape(xx.shape[0], 1))
yy = poly_linear_model_1.predict(xx_transformed_1)
plt.plot(xx, yy,label="$y = ax$")
plt.scatter(X_train, Y_train)
plt.scatter(X_test, Y_test, c = 'r')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a45ea9890>




![png](./Hengchao_01_25_1.png)


### weights of polynomial regression for order is 3


```python
xx = np.linspace(0, 1, 100)
xx_transformed_3 = polynomial_3.fit_transform(xx.reshape(xx.shape[0], 1))
yy = poly_linear_model_3.predict(xx_transformed_3)
plt.plot(xx, yy,label="$y = ax3+bx2+cx+d$")
plt.scatter(X_train, Y_train)
plt.scatter(X_test, Y_test, c = 'r')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a4606f6d0>




![png](./Hengchao_01_27_1.png)


### weights of polynomial regression for order is 9


```python
xx = np.linspace(0, 1, 100)
xx_transformed_9 = polynomial_9.fit_transform(xx.reshape(xx.shape[0], 1))
yy = poly_linear_model_9.predict(xx_transformed_9)
plt.plot(xx, yy,label="$y = ax9+....$")
plt.scatter(X_train, Y_train)
plt.scatter(X_test, Y_test, c = 'r')
plt.ylim(-1.5 ,1.5)
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a46153d90>




![png](./Hengchao_01_29_1.png)


## e. Draw train error vs test error


```python
train_error = [0]*10     #train error
test_error = [0]*10      #test error
```


```python
def getMse(Y, yy):
    standard = tf.square(Y - yy)
    mse = tf.reduce_mean(standard)
    return mse.numpy()
```


```python
def getError(i:int,  model) :
    polynomial = PolynomialFeatures(degree = i)
    xx_transformed_test = polynomial.fit_transform(X_test.reshape(X_test.shape[0], 1))
    xx_transformed_train = polynomial.fit_transform(X_train.reshape(X_test.shape[0], 1))
    yy_test = model.predict(xx_transformed_test)
    yy_train = model.predict(xx_transformed_train)

    test_error[i] = getMse(Y_test, yy_test)

    train_error[i] = getMse(Y_train, yy_train)
```


```python
polynomial_2, poly_linear_model_2 = polynomialRegression(2)
polynomial_4, poly_linear_model_4 = polynomialRegression(4)
polynomial_5, poly_linear_model_5 = polynomialRegression(5)
polynomial_6, poly_linear_model_6 = polynomialRegression(6)
polynomial_7, poly_linear_model_7 = polynomialRegression(7)
polynomial_8, poly_linear_model_8 = polynomialRegression(8)
# 0,1,3,9 I used the model fitted before.
getError(0, poly_linear_model_0)
getError(1, poly_linear_model_1)
getError(2, poly_linear_model_2)
getError(3, poly_linear_model_3)
getError(4, poly_linear_model_4)
getError(5, poly_linear_model_5)
getError(6, poly_linear_model_6)
getError(7, poly_linear_model_7)
getError(8, poly_linear_model_8)
getError(9, poly_linear_model_9)

print(test_error)
print(train_error)
```

    [0.725708887555328, 0.420433859602508, 0.4115158761946587, 0.03579509356913618, 0.038926156314548516, 0.033872153110566336, 0.05833211565743508, 0.1812106704430893, 0.554845693668926, 44.22283722466817]
    [0.3079871717563623, 0.24240685436800796, 0.23498551578658944, 0.01957321021379334, 0.01690032825391209, 0.005587179857868378, 0.003830224903521952, 0.000986326895842674, 0.0006297917682982702, 8.623765820671452e-21]



```python
xx = np.linspace(0, 9, 10)
plt.ylim(0 ,1)
plt.xlim(0,9)
plt.plot(xx, test_error, label = "$test error$", c = 'r')
plt.plot(xx, train_error, label = "$train error$", c = 'b')

plt.xlabel('Orders')

plt.ylabel('Error')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a4dd77a10>




![png](./Hengchao_01_35_1.png)


## f. Generate 100 more data and fit 9th order model and draw fit


```python
x_100 = np.linspace(0,1,100)     # Gegerate new 100 samples
d_100 = np.random.normal(loc=0,scale=0.2,size=100)    # N from the normal gaussian distribution 
y_100 = 2*math.pi*x_100
for i in range(100):
    y_100[i] = math.sin(y_100[i])+ d_100[i]
data_1 = {'X':x_100, 'Y':y_100}
data_100 = pd.DataFrame(data = data_1, dtype=np.int8)
data_100
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-0.033498</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.010101</td>
      <td>-0.028916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.020202</td>
      <td>0.021077</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.030303</td>
      <td>0.065874</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.040404</td>
      <td>0.301074</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.959596</td>
      <td>-0.192493</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.969697</td>
      <td>-0.360793</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.979798</td>
      <td>0.178334</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.989899</td>
      <td>-0.137827</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1.000000</td>
      <td>-0.270754</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>




```python
plt.scatter(x_100, y_100, marker = "o",c = "r")
```




    <matplotlib.collections.PathCollection at 0x1a491b45d0>




![png](./Hengchao_01_38_1.png)



```python
polynomial = PolynomialFeatures(degree = 9)# quadratic polynomial
x_transformed = polynomial.fit_transform(x_100.reshape(100,1))
poly_linear_model = LinearRegression()
poly_linear_model.fit(x_transformed, y_100)# train

xx_transformed_9 = polynomial.fit_transform(x_100.reshape(x_100.shape[0], 1))
yy = poly_linear_model.predict(xx_transformed_9)
plt.plot(x_100, yy,label="$y = ax9+.....$")
plt.scatter(x_100, y_100, c = "r")
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a4a4e5350>




![png](./Hengchao_01_39_1.png)


## g. Regularize using the sum of weights. 
## h. Draw chart for lamda



```python
def regularizeRidge(alpha):
    alpha = math.exp(alpha)
    print(alpha)
    polynomial = PolynomialFeatures(degree = 9)# quadratic polynomial
    x_transformed = polynomial.fit_transform(X_train.reshape(10,1))
    poly_linear_model = Ridge(alpha = alpha)
    poly_linear_model.fit(x_transformed, Y_train)# train
    
    return poly_linear_model

def chartRidge(alpha):
    model = regularizeRidge(alpha)
    xx = np.linspace(0, 1, 100)
    x_transformed = polynomial.fit_transform(xx.reshape(100,1))
    yy = model.predict(x_transformed)
    plt.plot(xx, yy,label=alpha)
    plt.scatter(X_train, Y_train)
    plt.scatter(X_test, Y_test, c = 'r')
    plt.legend()
```


```python
chartRidge(-35) #ln(lambda) = -35, lambda = 6.305116760146989e-16
```

    6.305116760146989e-16


    /Users/wanghengchao/anaconda3/envs/ML/lib/python3.7/site-packages/sklearn/linear_model/_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=6.22001e-17): result may not be accurate.
      overwrite_a=True).T



![png](./Hengchao_01_42_2.png)



```python
chartRidge(-30)    #ln(lambda) = -30, lambda = 9.357622968840175e-14
```

    9.357622968840175e-14



![png](./Hengchao_01_43_1.png)



```python
chartRidge(-25)    #ln(lambda) = -25, lambda = 1.3887943864964021e-11
```

    1.3887943864964021e-11



![png](./Hengchao_01_44_1.png)



```python
chartRidge(-20)    #ln(lambda) = -20, lambda = 2.061153622438558e-09
```

    2.061153622438558e-09



![png](./Hengchao_01_45_1.png)



```python
chartRidge(-15)    #ln(lambda) = -15, lambda = 3.059023205018258e-07
```

    3.059023205018258e-07



![png](./Hengchao_01_46_1.png)


## i. Draw test  and train error according to lamda 


```python
train_error_ridge = np.zeros(30)
test_error_ridge = np.zeros(30)

def getErrorRidge(i:int,  model) :     # A new error function
    xx_transformed_test = polynomial.fit_transform(X_test.reshape(X_test.shape[0], 1))
    xx_transformed_train = polynomial.fit_transform(X_train.reshape(X_train.shape[0], 1))
    yy_test = model.predict(xx_transformed_test)
    yy_train = model.predict(xx_transformed_train)
    test_error_ridge[i] = getMse(Y_test, yy_test)
    train_error_ridge[i] = getMse(Y_train, yy_train)
```


```python
xx = list(range(-30, 0))
for i in xx:
    model = regularizeRidge(i)
    getErrorRidge(i+30, model)
```

    9.357622968840175e-14
    2.543665647376923e-13
    6.914400106940203e-13
    1.8795288165390832e-12
    5.109089028063325e-12
    1.3887943864964021e-11
    3.775134544279098e-11
    1.026187963170189e-10
    2.7894680928689246e-10
    7.582560427911907e-10
    2.061153622438558e-09
    5.602796437537268e-09
    1.522997974471263e-08
    4.139937718785167e-08
    1.1253517471925912e-07
    3.059023205018258e-07
    8.315287191035679e-07
    2.2603294069810542e-06
    6.14421235332821e-06
    1.670170079024566e-05
    4.5399929762484854e-05
    0.00012340980408667956
    0.00033546262790251185
    0.0009118819655545162
    0.0024787521766663585
    0.006737946999085467
    0.01831563888873418
    0.049787068367863944
    0.1353352832366127
    0.36787944117144233



```python
xx = list(range(-30, 0))
plt.ylim(0 ,0.5)
plt.xlim(-30,0)
plt.plot(xx, test_error_ridge, label = "$test-error$", c = 'r')
plt.plot(xx, train_error_ridge, label = "$train-error$", c = 'b')

plt.xlabel('ln(lamdba)')

plt.ylabel('Error')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a4dfe1a50>




![png](./Hengchao_01_50_1.png)



```python
#   get the best lambda
best_lambda = 0
for i in range(-30,0):
    if test_error_ridge[i+30] == test_error_ridge.min(): best_lambda = i
print("best ln(lambda) = ", best_lambda)
best_lambda = math.exp(best_lambda)
print("best lambda = ", best_lambda)
```

    best ln(lambda) =  -13
    best lambda =  2.2603294069810542e-06


### In summary, the model which ln(lamdba) = -13, lambda = 2.2603294069810542e-06 has the best test performance. 
