x = [4,8,13,7]
y = [11,4,5,14]

xbar=0
ybar=0
summ=0

for i in x :
    summ = summ + i
xbar = summ/(len(x))
summ=0
for i in y :
    summ = summ + i

ybar = summ/(len(x))

xminbar = []
yminbar = []

for i in x :
    xminbar.append(i-xbar)

for i in y :
    yminbar.append(i-ybar)

xmininto=[]
for i in range(0,len(x)):
    xmininto.append(xminbar[i]*yminbar[i])

numerator=0
for i in xmininto:
    numerator = numerator + i

covxy=numerator/(len(x)-1)


xminbarsquare=[]
for i in xminbar:
    xminbarsquare.append(i*i)

yminbarsquare=[]
for i in yminbar:
    yminbarsquare.append(i*i)

xsquaresum=0
for i in xminbarsquare:
    xsquaresum = xsquaresum + i

ysquaresum=0
for i in yminbarsquare:
    ysquaresum = ysquaresum + i

varx = xsquaresum/(len(x)-1)
vary = ysquaresum/(len(x)-1)

import numpy as np

Sigma = np.array([[varx, covxy],
                  [covxy, vary]])

eigenvalues, eigenvectors = np.linalg.eig(Sigma)



pcpointsforeigen1=[]
for i in range(0,len(x)):
    pcpointsforeigen1.append((xminbar[i] * float(eigenvectors[0,1])) +
                (yminbar[i] * float(eigenvectors[0,0])))

pcpointsforeigen2=[]
for i in range(0,len(x)):
    pcpointsforeigen2.append((xminbar[i] * float(eigenvectors[1,1])) +
                (yminbar[i] * float(eigenvectors[1,0])))

print("PCA Points for Eigen Vector set 1 : ",pcpointsforeigen1)
print("PCA Points for Eigen Vector set 2 : ",pcpointsforeigen2)