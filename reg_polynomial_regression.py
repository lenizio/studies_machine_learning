import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Linear_Regression() : 
    
    def __init__( self, learning_rate, iterations, degrees, lambda_) : 
        
        self.learning_rate = learning_rate 
        self.iterations = iterations 
        self.cost_history = []
        self.parameters_history = []
        self.degrees= degrees
        self.lambda_= lambda_
                    
    def fit( self, X, Y ) : 
        
                
        self.b = 0
        self.X = self.x_transform(X, self.degrees)
        self.m, self.n = self.X.shape
        self.W = np.zeros(self.n)

        self.Y = Y 
              
        for i in range( self.iterations ) : 
            self.parameters_history.append([self.W, self.b])
            self.cost_history.append(self.compute_cost()) 
            self.update_weights() 
            
        return self
        
    def update_weights( self ) : 
            
        Y_pred = self.predict( self.X )     
        dW =( (( self.X.T ).dot(Y_pred - self.Y)) / self.m ) + ((self.lambda_ /self.m) * self.W)
        db = np.sum( Y_pred - self.Y) / self.m  
        
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db 
        
        return self
        
    def compute_cost(self):
            
        Y_pred= self.predict( self.X )
        cost = np.sum((Y_pred - self.Y)**2)/(2*self.m)
        
        reg_cost=(self.lambda_/2* self.m)*np.sum((self.W**2))    
        total_cost=cost+reg_cost
        
        return total_cost        
    
    def x_transform(self , X, degrees):
    
        t = X.copy().reshape(-1,1)
        X=X.reshape(-1,1)
    
        for i in degrees:
            X = np.append(X, t**i, axis=1)
            
        return X
        
    def predict( self, X ) : 
    
        return (X.dot(self.W))  + self.b 
    
def zscore_normalize_features(X):

    mu     = np.mean(X, axis=0)                
    sigma  = np.std(X, axis=0)                  
    X_norm = (X - mu) / sigma      

    return X_norm    

    
diabetes= pd.read_csv(r"/home/lenizio/datascience/diabetes_project/data/diabetes_filtered.csv")
    
X = zscore_normalize_features(diabetes.iloc[:,4].values)
X = X.astype(np.float128)
Y = diabetes.Glucose.values
    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )  
model = Linear_Regression( iterations = 1000, learning_rate = 0.01, degrees=[2,3])
poly_features = PolynomialFeatures(degree=[2,3])
model.fit(X_train,Y_train)

x_test= model.x_transform(X_test, [2,3])

y=model.predict(x_test)

plt.plot(np.arange(0,1000), model.cost_history)