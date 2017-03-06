# airline database

import pandas as pd
f=pd.read_csv("F:\PYTHON PROGRAMS\gitClone\machine_learning_examples-master\airline\international-airline-passengers.csv", engine="pyhton", skipfooter=3)

print(f.coloumns)

f.coloumns= ["months","passengers"]

f['ones']=1
print(f.head())
