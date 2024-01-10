import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


class Logistic_Regression() :
    
    def __init__( self, learning_rate, iterations, lambda_) : 
        
        self.learning_rate = learning_rate 
        self.iterations = iterations 
        self.cost_history = []
        self.parameters_history = []
        self.lambda_= lambda_
                    
    def fit( self, X, Y ) : 
        
                
        self.X = X
        self.Y = Y 
        self.m, self.n = self.X.shape
        self.W = np.zeros(self.n)
        self.b = 0
              
        for i in range( self.iterations ) : 
            self.parameters_history.append([self.W, self.b])
            self.cost_history.append(self.compute_cost()) 
            self.update_weights() 
            
        return self
        
    def update_weights( self ) : 
            
        Y_pred = self.calculate_fwb( self.X )     
        dW =( (( self.X.T ).dot(Y_pred - self.Y)) / self.m ) + ((self.lambda_ /self.m) * self.W)
        db = np.sum( Y_pred - self.Y) / self.m  
        
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db 
        
        return self
        
    def compute_cost(self):
            
        Y_pred = self.calculate_fwb( self.X )
        cost = np.mean(-self.Y * np.log(Y_pred) - (1-self.Y) * np.log(1-Y_pred))
        
        reg_cost=(self.lambda_/(2* self.m))*np.sum((self.W**2))    
        total_cost=cost+reg_cost
        
        return total_cost        
    
    def calculate_fwb(self, X):
        
        fwb = X.dot(self.W) + self.b
        sigmoid = 1 / (1 + np.exp(-fwb))
        return sigmoid 
            
    def predict( self, X ) : 
        
        pred = self.calculate_fwb(X)
        pred = np.where(pred>=0.2, 1, 0)
    
        return pred 
    

def zscore_normalize_features(X):

    mu     = np.mean(X, axis=0)                
    sigma  = np.std(X, axis=0)                  
    X_norm = (X - mu) / sigma      

    return X_norm    

    




