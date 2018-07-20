import numpy as np
from sklearn.tree import DecisionTreeRegressor

N = 100
p = 10
sigmax = 2
X = np.zeros((N,p))

for j in range(0, p):
   X[..., j] = sigmax * np.random.randn(N)

sigmay = 1
ns = 10  # number of simulations
dof = np.zeros(ns)

for j in range(0, ns-1):
  y = sigmay * np.random.randn(N).T

  #regr = DecisionTreeRegressor(max_depth=5)
  #regr = DecisionTreeRegressor(min_samples_split=10)
  regr = DecisionTreeRegressor(min_samples_leaf=16)  #100, 16, 8
  regr.fit(X, y)

  y_pred = regr.predict(X)
  [y_uniq,counts] = np.unique(y_pred, return_counts='true')
  print np.size( y_uniq ) 
  print counts
    
  dof[j] =  counts / N * sigmay**2 
    
  ##M = np.row_stack((y,y_pred))
  ##cov_pred = np.cov(M)
  #cov_pred = np.cov(y,y_pred)
  #dof[j] = np.trace( cov_pred ) / sigmay**2 # degree of freedom of a single simulation
  
  #print cov_pred
  #print dof[j]

#avg_dof = np.mean(dof)  # average degree of freedom
#print avg_dof
