# Copyright: Shaohao Chen, Research Computing Services, Boston University. 2018

# Import necessary python modules.
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib import colors

# A function to construct quadratic basis from linear basis
def quad_basis(h_lin):
    dim = h_lin.shape[1]  # number of original dimensions = number of columes of X
    dim_add = dim*(dim+1)/2  # number of dimensions to be added
    n_row = h_lin.shape[0]   # number of rows of X
    h_add = np.zeros((n_row,dim_add))  # shape of the sub matrix for new dimensions
    index=0
    for i in range(0, dim):
        for j in range(i, dim):
           h_add[...,index] = h_lin[...,i]*h_lin[...,j]  # compute the sub matrix for new dimensions
           index += 1
    h_quad = np.concatenate((h_lin, h_add), axis=1)   # append new dimensions to obtain new X
    return h_quad

# A function to plot results
def plot_classification(y, y_pred, dim1, dim2, p_dim, xlab, ylab, title, n_fig):
    msize=8
    mcolor=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    correct_i = np.where(y==y_pred)   # Index of correct predictions
    error_i = np.where(y!=y_pred)     # Index of error predictions
    for i in range(p_dim):
       class_i = np.where(y==i+1)  # Index of the i-th class
       correct_class_i = np.intersect1d(correct_i, class_i)  # Index of correct predictions in the i-th class
       error_class_i = np.intersect1d(error_i, class_i)   # Index of error predictions in the i-th class
       plt.plot(X0[correct_class_i, dim1], X0[correct_class_i, dim2], 'o', markeredgecolor=mcolor[i],
markerfacecolor='None', markersize=msize)   # plot correct predictions
       plt.plot(X0[error_class_i, dim1], X0[error_class_i, dim2], '^', markeredgecolor=mcolor[i], markersize=msize)   # plot error predictions
    lsize=18
    plt.xlabel(xlab, fontsize=lsize)
    plt.ylabel(ylab, fontsize=lsize)
    tsize=20
    full_title = "Fig." + n_fig + "  " + title
    plt.title(full_title, fontsize=tsize)

# ====== Start main program ======
# Read vovwel data from csv file and save in numpy arrays
training_data = np.genfromtxt('vowel_data/training_nohead.csv', delimiter=',')
test_data = np.genfromtxt('vowel_data/test_nohead.csv', delimiter=',')

# Feed X and y with training data
y = training_data[...,1]  # 2nd column
X = training_data[...,2:]  # from 3rd column onwards
# Feed X0 and y0 with test data
y0 = test_data[...,1]  # 2nd column
X0 = test_data[...,2:]  # from 3rd column onwards

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)  # Call QDA in Scikit-learn
qda.fit(X, y)    # Fit with training data
y0_qda = qda.predict(X0)  # Predict with test data
print "====== QDA ======"
print "Misclassification rate of training data = ", 1 - qda.score(X, y)   
print "Misclassification rate of test data = ", 1 - qda.score(X0, y0)  

# Construct quadratic basis from linear basis for both training and test data
X_quad = quad_basis(X)
X0_quad = quad_basis(X0)
# Linear Discriminant Analysis with quadratic basis
lda_quad = LinearDiscriminantAnalysis(solver="lsqr",store_covariance=True) # Call LDA in Scikit-learn
lda_quad.fit(X_quad, y)    # Fit with training data
y0_lda_quad = lda_quad.predict(X0_quad)  # Predict with test data
print "====== LDA: quadratic basis ======"
print "Misclassification rate of training data = ", 1 - lda_quad.score(X_quad, y) 
print "Misclassification rate of test data = ", 1 - lda_quad.score(X0_quad, y0)   

# Visialization
width = 12
height = 10
plt.figure(figsize=(width, height))
plot_classification(y0, y0_qda, 0, 1, 10, "x1", "x2", "QDA results in subspace (x1, x2)", "1")
plt.figure(figsize=(width, height))
plot_classification(y0, y0_qda, 4, 5, 10, "x5", "x6", "QDA results in subspace (x5, x6)","2")
plt.figure(figsize=(width, height))
plot_classification(y0, y0_qda, 8, 9, 10, "x9", "x10", "QDA results in subspace (x9, x10)","3")
plt.figure(figsize=(width, height))
plot_classification(y0, y0_lda_quad, 0, 1, 10, "x1", "x2", "LDA results in subspace (x1, x2)","4")
plt.figure(figsize=(width, height))
plot_classification(y0, y0_lda_quad, 4, 5, 10, "x5", "x6", "LDA results in subspace (x5, x6)","5")
plt.figure(figsize=(width, height))
plot_classification(y0, y0_lda_quad, 8, 9, 10, "x9", "x10", "LDA results in subspace (x9, x10)","6")
plt.show()
