# -*- coding: utf-8 -*-
"""This file contains an implementation of the NNQF filter presented in:
    
González Ordiano, J. Á., Gröll, L., Mikut, R., & Hagenmeyer, V. (2020). 
Probabilistic energy forecasting using the nearest neighbors quantile filter 
and quantile regression. International Journal of Forecasting, 36(2), 310-323.


MIT License

Copyright (c) 2020 Jorge Ángel González Ordiano, jorge.gonzalez@ibero.mx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


"""
#Libraries
import numpy as np
from sklearn.neighbors import NearestNeighbors as nn_fun


#----
#Functions
def nnqf_filter(x_input,y_output,num_neighbors = 10,q_quantile = 0.5,var_weighting = True,minkowski_dist = 2):
   """
   Parameters
   ----------
   x_input : numpy array ;
   Input matrix of dimension (N,S), with N representing the number of
   samples and S the number of features
    
   y_output : numpy array ;
   Output vector of dimension (N,)
              
   num_neighbors : int, default = 10 ;
   Number of nearest neighbors that the filter is going to search for     
   
   q_quantile : float, default = 0.5 ;
   Must be a value between 0 and 1. 
   Probability of the quantile that is going to be calculated from the 
   nearest neighbors output values      
   
   var_weighting : bool, default = True ;
   Value definign if the columns of the input matrix are going to be multiplied 
   by the inverse of their variance
   
   minkowski_dist : int, default = 2 ;
   Parameter used to define the type of minkoswki distance used to calculate 
   the nearest neighbors
     
   Returns
   -------
   yq_output : numpy array ;
   New output vector containing the quantiles of the output values of the
   input's nearest neighbors
      
   """
   
   #--
   #Each column of the input matrix is multiplied by the inverse of its variance, 
   #in order to avoid a feature with a huge scale to overpower the others at the 
   #moment of caluclating the distances
    
   if var_weighting:
       var_weights = np.var(x_input,axis = 0)
       x_input = var_weights**(-1) * x_input    
    
   #--
   #We calculate the nearest neighbor of each feature vector within the input matrix
   #and obtain their corresponding indices
   #The distance used is the minkowski distance with p = minkowski_dist

   x_neighbors = nn_fun(n_neighbors=num_neighbors,algorithm='auto',p=minkowski_dist).fit(x_input)
   dist, indx = x_neighbors.kneighbors(x_input)
   
    
    #--
    #We create a matrix containing the output values of neareast neighbors of
    #each input vector
    
   y_neighbors = y_output[indx[0,:]]
   for i in range(1, np.size(x_input,0)):
       values_to_add = y_output[indx[i,:]]
       y_neighbors = np.vstack([y_neighbors,values_to_add])

   #--
   #We calculate the q_quantile of the of the nearest neighbors output values
   #and create with them a new output vector yq_output
    
   yq_output = np.quantile(y_neighbors,q=q_quantile,axis=1)
       
   return yq_output
    

#----
#Example

x_input = np.array([[5,3,4,6,7,8,2], 
                    [6,3,2,5,1,1,0], 
                    [7,7,8,3,5,2,9], 
                    [4,5,7,2,4,6,6], 
                    [0,1,2,4,4,5,9], 
                    [4,2,5,6,1,3,2],
                    [7,4,8,1,9,5,3],
                    [9,7,0,3,5,6,3],
                    [8,3,7,3,9,1,1],
                    [3,6,7,3,9,0,4]])


y_output = np.array([1,5,7,3,8,2,4,6,8,4])

#y_output = []

yq_output = nnqf_filter(x_input, y_output, num_neighbors = 3)