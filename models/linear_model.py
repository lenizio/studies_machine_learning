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
                
        self.m = X.shape[0]
        self.W = 0
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
        dW = ( self.X.T ).dot(Y_pred - self.Y) / self.m 
        db = np.sum( Y_pred - self.Y) / self.m  
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db 
        
        return self
        
    def compute_cost(self):
            
        Y_pred= self.predict( self.X )
        cost = np.sum((Y_pred - self.Y)**2)/(2*self.m)  
        
        return cost        
        
    def predict( self, X ) : 
    
        return X * self.W  + self.b 
    
def zscore_normalize_features(X):

    mu     = np.mean(X, axis=0)                
    sigma  = np.std(X, axis=0)                  
    X_norm = (X - mu) / sigma      

    return X_norm    

    
diabetes= pd.read_csv(r"/home/lenizio/datascience/diabetes_project/data/diabetes_filtered.csv")
X = X = zscore_normalize_features(diabetes.Insulin.values)
Y = diabetes.Glucose.values
    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )
    
model = Linear_Regression( iterations = 1000, learning_rate = 0.01 ) 

Y_predc=model.predict(X_test)

fig, ax =  plt.subplots()
ax.scatter(X_train,Y_train)
ax.plot(X_test,Y_predc, color="r")


reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_sklearn = reg.predict(X_test.reshape(-1,1))

fig, ax =  plt.subplots()
ax.scatter(X_train,Y_train)
ax.plot(X_test,Y_sklearn, color="r")