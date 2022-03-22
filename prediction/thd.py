from matplotlib.pyplot import axis
import pandas as pd
import numpy as np

inputframe = pd.read_csv('predictions.csv')

columnnames = [ 'targetTHD', 'actualTHD', 'variance' ]

outputframe =  pd.DataFrame(columns = columnnames)

targetTHDs = inputframe['thd']
outputframe['targetTHD'] = targetTHDs

squareframe = inputframe.drop(['thd', 'frequency', 'h1'], axis=1)
squareframe = squareframe **2
squares = squareframe.sum(axis=1)
roots = np.sqrt(squares)
v1 = inputframe.at[1,'h1']
actualTHDs = roots/v1*100

variances = (actualTHDs-targetTHDs)/targetTHDs*100

outputframe['actualTHD'] = actualTHDs
outputframe['variance'] = variances

outputframe.to_csv('variance.csv', index=False)