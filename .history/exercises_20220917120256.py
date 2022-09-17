# Import packages
from statsmodels.formula.api import ols
import pandas
import scipy
from scipy import stats
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

data = pandas.read_csv('data/brain_size.csv', sep=';', na_values='.')

##################
##  Exercise 1  ##
##################

# What is the mean value for VIQ for the full population?
data[['VIQ']].mean()

# How many males/females were included in this study?
groupby_gender = data.groupby('Gender')
groupby_gender.count()

# What is the average value of MRI counts expressed in log units, for males and females?


##################
##  Exercise 2  ##
##################
