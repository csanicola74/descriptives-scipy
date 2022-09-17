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
data['MRI_base10'] = np.log10(data['MRI_Count'])
print(data['MRI_base10'])

##################
##  Exercise 2  ##
##################

# Plot the scatter matrix for males only, and for females only.
scatter_matrix(data[['VIQ', 'MRI_Count', 'Height']],
               c=(data['Gender'] == 'Female'), marker='o',
               alpha=1, cmap='winter')

fig = plt.gcf()
fig.suptitle("blue: male, green: female", size=13)

plt.show()
# Do you think that the 2 sub-populations correspond to gender?
# Yes, I believe that the 2 sub-populations correspond to gender

##################
##  Exercise 3  ##
##################