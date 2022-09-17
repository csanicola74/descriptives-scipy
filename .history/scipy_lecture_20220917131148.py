# Import packages
from statsmodels.formula.api import ols
import pandas
import scipy
from scipy import stats
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

###############################
##  Reading from a CSV file  ##
###############################

data = pandas.read_csv('data/brain_size.csv', sep=';', na_values='.')
data

############################
##  Creating from arrays  ##
############################


# create the 3 numpy arrays
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

# then expose them as a dataframe
pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

#########################
##  Manipulating data  ##
#########################

# data is a pandas.DataFrame that resembles R's dataframe

data.shape  # 40 rows and 8 columns
data.columns    # it has columns
print(data['Gender'])   # Columns can be addressed by name

# simpler selector
# this is finding all the females and then getting the mean of the VIQ values for them
data[data['Gender'] == 'Female']['VIQ'].mean()

# groupby: splitting a dataframe on values of categorical variables
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))

groupby_gender.mean()   # this will now show the means for each column by gender

#####################
##  Plotting data  ##
#####################

# pandas comes with some plotting tools to display statistics of the data in dataframes:
scatter_matrix(data, [['Weight', 'Height', 'MRI_Count']])
scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])

################################################
##  Hypothesis Testing: Comparing two groups  ##
################################################

# 1-sample t-test: testing the value of a population mean
stats.ttest_1samp(data['VIQ'], 0)

# 2-sample t-test: testing for difference across populations
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)

###############################################################
##  Paired Tests: Repeated Measures on the Same Individuals  ##
###############################################################

stats.ttest_ind(data['FSIQ'], data['PIQ'])
# FSQIP and PIQ are measured on the same individuals
# the variance due to inter-subject variability is confounding, and can be removed, using a "paired test", or "repeated measures test"
stats.ttest_rel(data['FSIQ'], data['PIQ'])
# this is equivalent to a 1-sample test on the difference
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)
# T-tests assume Gaussian errors
# we can use a Wilcoxon signed-rank test that relaxes this assumption
stats.wilcoxon(data['FSIQ'], data['PIQ'])
# Note: the corresponding test in the non paired case is the Mann-Whitney U test
# scipy.stats.mannwhitneyu()

#################################################################
##  Linear models, multiple factors, and analysis of variance  ##
#################################################################

# A simple linear regression
# first, generate simulated data according to the model
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y})

# then specify an OLS model and fit it
model = ols("y ~ x", data).fit()
print(model.summary())
# y (endogenous) is the value you are trying to predict
# x (exogenous) represents the features you are using to make the prediction

######################################################################
##  Categorical variables: comparing groups or multiple categories  ##
######################################################################

# lets us go back to the data on brain size:
data = pandas.read_csv('data/brain_size.csv', sep=';', na_values='.')

# we can write a comparison between IQ of male and female using a linear model:
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())

# TIPS
# Forcing categorical: the ‘Gender’ is automatically detected as a categorical variable, and thus each of its different values are treated as different entities.
# Intercept: We can remove the intercept using - 1 in the formula, or force the use of an intercept using + 1.

######################################################
##  Link to t-tests between different FSIQ and PIQ  ##
######################################################

data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long)
