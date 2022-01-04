#         Pima Indians diabetes database descriptive statistics project



```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
```

###                                                  1)        Basic exploration:-


```python
dbs = pd.read_csv(R"C:\Users\Administrator\Downloads\diabetes.csv")
dbs
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101</td>
      <td>76</td>
      <td>48</td>
      <td>180</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121</td>
      <td>72</td>
      <td>23</td>
      <td>112</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93</td>
      <td>70</td>
      <td>31</td>
      <td>0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 9 columns</p>
</div>




```python
dbs.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
dbs.describe(include="all")
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dbs.isnull().sum()
```




    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64



### Inference:- 
     1)There are no missing values in the database.
     2)There are 768 rows & 9 columns in the database.
     3)All datapoints are numerical data having both integer & float type datapoints.

### 2) Measures of central tendency for glucose and outcome column:-


```python
dbs["Glucose"].mean()
```




    120.89453125




```python
dbs["Glucose"].median()
```




    117.0




```python
dbs["Outcome"].mean()
```




    0.3489583333333333




```python
dbs["Outcome"].median()
```




    0.0



### inference:-
1) for glucose column- since mean & median are quite closer,we can say that it has symmetrical data points.
2) for outcome column- since mean & median are quie closer, it also has symmetrical data points.

### 3) Data summaries for all cloumns:-


```python
plt.figure(figsize=(15,5))
plt.title("pregnacy vs insulin levels")
plt.scatter(dbs["Pregnancies"],dbs["Insulin"])
plt.show()
```


    
![png](output_15_0.png)
    


#### Result 1) females whose insulin levels are (0-300) mu U/ml are more likely to get pregnant 7 times in their lifetime.


```python
plt.figure(figsize=(10,10))
plt.title("pregnacy vs blood pressure")
plt.scatter(dbs["Pregnancies"],dbs["BloodPressure"],color="red")
plt.show()
```


    
![png](output_17_0.png)
    


#### Result 2) females whose blood pressure is between(55-90)mm Hg are most likely to get pregnant 10times in                          their life time.


```python
plt.figure(figsize=(10,10))
plt.title("pregnacy vs age")
plt.scatter(dbs["Pregnancies"],dbs["Age"],color= "green")
plt.show()
```


    
![png](output_19_0.png)
    


#### Result 3) females whose age is between (21-33)years, are most likely to get pregnant 4 times , & females whose age is between (22-42) years, are likely to get pregnant 6 times in their lifetime.


```python
plt.figure(figsize=(8,8))
plt.title("BMI vs bloodpressure")
plt.scatter(dbs["BMI"],dbs["BloodPressure"],color= "green")
plt.show()
```


    
![png](output_21_0.png)
    


#### Result 4) Females whose BMI is between(20-45) are highly likely to have blood pressure in the range( 50-95) mm Hg and therefore highly likely to get pregnant 10 times in their life time.


```python
plt.figure(figsize=(15,10))
plt.title("Glucose vs BloodPressure")
plt.xlabel("Glucose")
plt.ylabel("Blood pressure")
plt.scatter(dbs["Glucose"],dbs["BloodPressure"],color= "cyan")
plt.show()
```


    
![png](output_23_0.png)
    


#### Result 5) Females whose glucose levels are between (50-90) are most likely to have blood pressure in the range (80-150)mm Hg.

### 4) Relationship between Age v/s Glucose


```python
plt.figure(figsize=(15,10))
plt.title("relationship between Age and Glucose")
sns.regplot(x="Age",y="Glucose",data=dbs)
plt.show()

```


    
![png](output_26_0.png)
    


#### result:- females of age group between (20-32) are having glucose level ranging from (80-140).

### 5) examining data based on skewness & normally distributed:-


```python
plt.figure(figsize=(15,10))
sns.displot(x="Pregnancies",data=dbs,color="blue",bins=10,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_29_1.png)
    



```python
plt.figure(figsize=(15,10))
sns.displot(x="Glucose",data=dbs,color="cyan",bins=10,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_30_1.png)
    



```python
plt.figure(figsize=(15,10))
sns.displot(x="BloodPressure",data=dbs,color="red",bins=10,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_31_1.png)
    



```python
plt.figure(figsize=(15,10))
sns.displot(x="SkinThickness",data=dbs,color="green",bins=20,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_32_1.png)
    



```python
plt.figure(figsize=(15,10))
sns.displot(x="Insulin",data=dbs,color="blue",bins=10,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_33_1.png)
    



```python
plt.figure(figsize=(15,10))
sns.displot(x="BMI",data=dbs,color="green",bins=10,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_34_1.png)
    



```python
plt.figure(figsize=(15,10))
sns.displot(x="DiabetesPedigreeFunction",data=dbs,color="orange",bins=10,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_35_1.png)
    



```python
plt.figure(figsize=(15,10))
sns.displot(x="Age",data=dbs,color="pink",bins=20,kde=True)
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_36_1.png)
    


#### result:-   1)glocose,blood pressure,skin thickness, and BMI are the numerical columns with normal distribution. 

#### 2)pregnancies,insulin,age and DiabetesPedigreeFunction are the numerical columns with right skwed data points.


### 6)Plot to visualize the distribution of Outcome variable


```python
plt.figure(figsize=(10,5))
ab=sns.countplot(data=dbs,x="Outcome").set(title='total count of diabetics & non-diabetics',xlabel='diabetics v/s non-diabetics',ylabel='count')
plt.show()
```


    
![png](output_40_0.png)
    


#### Inference:- non-diabetics are twice the number of diabetics.

### 7)calculation of the skewness value of each variable and segregating them accordingly:-


```python
Pregnancy=dbs["Pregnancies"]
Pregnancy.skew()
```




    0.9016739791518588




```python
Glucose=dbs["Glucose"]
Glucose.skew()
```




    0.17375350179188992




```python
BloodPressure=dbs["BloodPressure"]
BloodPressure.skew()
```




    -1.8436079833551302




```python
SkinThickness=dbs["SkinThickness"]
SkinThickness.skew()
```




    0.10937249648187608




```python
Insulin=dbs["Insulin"]
Insulin.skew()
```




    2.272250858431574




```python
BMI=dbs["BMI"]
BMI.skew()
```




    -0.42898158845356543




```python
DPF=dbs["DiabetesPedigreeFunction"]
DPF.skew()
```




    1.919911066307204




```python
Age=dbs["Age"]
Age.skew()
```




    1.1295967011444805




```python
Outcome=dbs["Outcome"]
Outcome.skew()
```




    0.635016643444986



#### scale of skewness:-
1) between -0.5 to 0.5== symmetrical
2) between (-1 & 0.5) and (0.5 & 1)== moderatly skewed
3) between (less than -1) and ( more than +1)== highly skewed

### Result:- 
1) pregnancies= moderately skewed
2) glucose= moderately skewed
3) blood pressure= highly skewed
4) SkinThickness= symmetrical
5) Insulin= highly skewed
6) BMI = symmetrical
7) DiabetesPedigreeFunction= highly skewed
8) Age = highly skewed
9) Outcome= moderately skewed


### 8) Boxplot to identify outliers for each variables of the data set


```python
sns.boxplot(x="Pregnancies",data=dbs)
```




    <AxesSubplot:xlabel='Pregnancies'>




    
![png](output_55_1.png)
    



```python
sns.boxplot(x="Glucose",color="green",data=dbs)
```




    <AxesSubplot:xlabel='Glucose'>




    
![png](output_56_1.png)
    



```python
sns.boxplot(x="BloodPressure",color="cyan",data=dbs)
```




    <AxesSubplot:xlabel='BloodPressure'>




    
![png](output_57_1.png)
    



```python
sns.boxplot(x="SkinThickness",color="pink",data=dbs)
```




    <AxesSubplot:xlabel='SkinThickness'>




    
![png](output_58_1.png)
    



```python
sns.boxplot(x="Insulin",color="brown",data=dbs)
```




    <AxesSubplot:xlabel='Insulin'>




    
![png](output_59_1.png)
    



```python
sns.boxplot(x="BMI",color="blue",data=dbs)
```




    <AxesSubplot:xlabel='BMI'>




    
![png](output_60_1.png)
    



```python
sns.boxplot(x="DiabetesPedigreeFunction",color="red",data=dbs)
```




    <AxesSubplot:xlabel='DiabetesPedigreeFunction'>




    
![png](output_61_1.png)
    



```python
sns.boxplot(x="Age",color="magenta",data=dbs)
```




    <AxesSubplot:xlabel='Age'>




    
![png](output_62_1.png)
    


#### Result:- All the variables have outliers.

### 9) Based on the above analysis, all the variables have outliers, hence the best measure of central tendency for all the variables is- MEDIAN.and the best measure of dispersion is - IQR( inter quartile range).


```python

```
