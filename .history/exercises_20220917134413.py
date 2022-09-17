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
data.columns
data.dtypes
data['Weight']
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

# Test the difference between weights in males and females.
stats.ttest_1samp(data.dropna()['Weight'], 0)
female_wt = data.dropna()[data['Gender'] == 'Female']['Weight']
male_wt = data.dropna()[data['Gender'] == 'Male']['Weight']
stats.ttest_ind(female_wt, male_wt)

# Use non parametric statistics to test the difference between VIQ in males and females.
female_viq = data.dropna()[data['Gender'] == 'Female']['VIQ']
male_viq = data.dropna()[data['Gender'] == 'Male']['VIQ']
scipy.stats.mannwhitneyu(female_viq, male_viq)

##################
##  Exercise 4  ##
##################

x = np.linspace(-5, 5, 20)
np.random.seed(1)
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
data = pandas.DataFrame({'x': x, 'y': y})

# Retrieve the estimated parameters from the model above.
model = ols("y ~ x", data).fit()
print(model.summary())

##################
##  Exercise 5  ##
##################
#
# Going back to the brain size + IQ data
# test if the VIQ of male and female are different after removing the effect of brain size, height and weight.


data = pandas.read_csv('../brain_size.csv', sep=';', na_values='.')
model = ols('VIQ ~ Gender + MRI_Count + Height', data).fit()
print(model.summary())
print(model.f_test([0, 1, 0, 0]))
