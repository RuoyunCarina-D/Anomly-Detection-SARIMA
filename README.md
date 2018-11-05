# Anomly Detection
## SARIMA

Applied SARIMA time series algorithm to predict sales amount based on past 4 year data and creat alerts if data is out of accepted range.

**SARIMA** stands for Seasonal (S) Auto-Regressive (AR) Integrated (I) Moving Average (MA). The Seasonal ARIMA model is to difference the series to make it stationary by taking differences of the variable over time. There are 3 methods to take differences in an ARIMA model - AR, I, and MA. 


**_I_** is a full difference. That is today’s value minus yesterday’s value.

**_AR_** is very much related. The way that I like to think of the AR term is that it is a partial difference. The coefficient on the AR term will tell you the percent of a difference you need to take. 

**_MA_** is what percent to add back into the error term after differencing. Greene’s book Econometric Analysis, makes the point that this is really about patching up the standard errors. At least that’s what I got out of his discussion of the topic.

### Packages used: 
* numpy,pandas
* statsmodels
* matplotlib
* plotly
* itertools
* sklearn


Jupyter notebook [here](https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA%20.ipynb)

If you intend to check interactive charts in the .ipynb, please check [here](http://nbviewer.jupyter.org/github/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/interactive%20source.ipynb) (Needed to login in Plotly account)

Anomaly detection of sales based on prediction done by SARIMA
