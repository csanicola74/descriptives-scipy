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
data.mean('VIQ')
# How many males/females were included in this study?
