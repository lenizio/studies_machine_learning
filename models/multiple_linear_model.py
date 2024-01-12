import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


class Linear_Regression() : 
    
    def __init__( self, learning_rate, iterations ) : 
        
        self.learning_rate = learning_rate 
        self.iterations = iterations 
        self.cost_history = []
        self.parameters_history = []
                    
    def fit( self, X, Y ) : 
                
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X 
        self.Y = Y 
              
        for i in range( self.iterations ) : 
            self.parameters_history.append([self.W, self.b])
            self.cost_history.append(self.compute_cost()) 
            self.update_weights() 
            
        return self
        
    def update_weights( self ) : 
            
        Y_pred = self.predict( self.X )     
        dW = (( self.X.T ).dot(Y_pred - self.Y)) / self.m 
        db = np.sum( Y_pred - self.Y) / self.m  
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db 
        
        return self
        
    def compute_cost(self):
            
        Y_pred= self.predict( self.X )
        cost = np.sum((Y_pred - self.Y)**2)/(2*self.m)  
        
        return cost        
        
    def predict( self, X ) : 
    
        return (X.dot(self.W))  + self.b 
    
def zscore_normalize_features(X):

    mu     = np.mean(X, axis=0)                
    sigma  = np.std(X, axis=0)                  
    X_norm = (X - mu) / sigma      

    return X_norm    

    

    
