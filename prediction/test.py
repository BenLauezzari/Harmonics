import pandas as pd
import numpy as np

inputframes = {}

for n in range(3, 30, 2):
    filename = '~/Documents/Documents/work/FARS/StandardShunt/F662/extrapolation/h' + str(n) + '.csv'
    inputframes[n] = pd.read_csv(filename, names=['index', 'v'] )

columnnames = [ 'thd', 'frequency' ]

for n in range(1, 51, 1):
    columnnames.append('h' + str(n) )

out = pd.DataFrame(columns = columnnames)

index = inputframes.get(3).get('index').to_numpy()
thd = []
for i in index:
    thd.append(i/2)

out['thd'] = thd
out['frequency'] = 50
out['h1'] = 119.98011800039

#put 0 in all the even harmonics
for n in range(2,51,2):
    col = 'h' + str(n)
    out[col] = 0

#transfer predictions to odd harmonics 3 to 29
for n in range(3, 30, 2):
    col = 'h' + str(n)
    out[col] = inputframes.get(n).get('v')
    out.loc[out[col] < 0, col] = 0  #subzero values become 0

#zero the rest of the odd columns
for n in range(31, 51, 2):
    col = 'h' + str(n)
    out[col] = 0

outfile = 'extrapolation.csv'
out.to_csv(outfile, index=False)

print(out)
