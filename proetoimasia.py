import pandas as pd
import numpy as np

features=['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']
classes=['sitting', 'sittingdown', 'standing', 'standingup', 'walking']
df = pd.read_csv('dataset-HAR-PUC-Rio.csv', delimiter=';', low_memory=False)



#vriskoume tous mesous orous
mv={}
stdv={}
for x in features:
    mv[x]=np.mean(df[x])
    stdv[x]=np.std(df[x])
    print(x,mv[x],stdv[x])

df2=df
# filtraroume me vasi tin mesi timi kai tipiki apoklisi
for x in features:
    df3=df2[(df2[x]>=mv[x]-2*stdv[x]) & (df2[x]<=mv[x]+2*stdv[x]) ]
    df2=df3

#pairnoume ta nea mesi timi, stdev, max, min
mv={}
stdv={}
maxv={}
minv={}
for x in features:
    mv[x]=np.mean(df2[x])
    stdv[x]=np.std(df2[x])
    maxv[x]=np.max(df2[x])
    minv[x]=np.min(df2[x])
    
    print(x,mv[x],stdv[x], maxv[x],minv[x])


#kanoume kanonikopoiisi

for x in features:
    df2[x]=(df[x]-minv[x])/(maxv[x]-minv[x])

for x in features:
    print(x,np.mean(df2[x]))
    
#gia kathe klasi exoume
M={}
for y in classes:
    dfc=df2[df2['class']==y]
    mvc={}
    for x in features:
        mvc[x]=np.mean(dfc[x])
    M[y]=mvc
    
# epilogi tou arxikou plithismou


dfsitting=df2[df2['class']=='sitting']
dfsittingvector=dfsitting[features]
dfsittingvector.to_csv('data_sitting.csv', index=False)



