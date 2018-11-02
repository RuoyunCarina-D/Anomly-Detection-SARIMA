# Anomly-Detection-SARIMA

Applied SARIMA time series algorithm to predict sales amount based on past 4 year data and creat alerts if data is out of accepted range.

stands for Auto-Regressive (AR) Integrated (I) Moving Average (MA). The ARIMA model is all about differencing the series to make it stationary. We do this by taking differences of the variable over time. We have three methods of “taking differences” available to us in an ARIMA model. The AR term, the I term, and the MA term. The $I$ term is a full difference. That is today’s value minus yesterday’s value. That’s it.The $AR$ term is very much related. The way that I like to think of the AR term is that it is a partial difference. The coefficient on the AR term will tell you the percent of a difference you need to take. $MA$ term tells you what percent to add back into the error term after differencing. Greene’s book Econometric Analysis, makes the point that this is really about patching up the standard errors. At least that’s what I got out of his discussion of the topic.


* numpy,pandas
* statsmodels
* matplotlib
* plotly
* itertools
* sklearn


Jupyter notebook [here](https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA%20.ipynb)

If you intend to check interactive charts in the .ipynb, please check [here](http://nbviewer.jupyter.org/github/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/interactive%20source.ipynb) (Needed to login in Plotly account)

Anomaly detection of sales based on prediction done by SARIMA
