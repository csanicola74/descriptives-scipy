# Import packages
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
px.scatter_matrix(data,
                  dimensions=['Weight', 'Height', 'MRI_Count'])
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