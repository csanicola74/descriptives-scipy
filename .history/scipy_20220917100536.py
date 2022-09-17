import matplotlib.pyplot as plt
from pandas.plotting import plotting
import numpy as np
from termios import TAB0
import pandas

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

########################
##  Manipulating data ##
########################

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

####################
##  Plotting data ##
####################

# pandas comes with some plotting tools to display statistics of the data in dataframes:
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
