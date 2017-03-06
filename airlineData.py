# airline database

import pandas as pd
f=pd.read_csv("international-airline-passengers.csv", engine="pyhton", skipfooter=3)

print(f.coloumns)

#renaming the coloumns as the name is not good

f.coloumns= ["months","passengers"]

#adding an extra coloumns with only 1's

f['ones']=1
print(f.head())
