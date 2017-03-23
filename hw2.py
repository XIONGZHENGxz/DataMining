from sklearn import datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold,train_test_split
X = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=20)
from sklearn.linear_model import Lasso,Ridge
from sklearn import metrics
import numpy as np
alphas =  10**np.linspace(-2,10,100)*0.5
print alphas
#function for k fold cross validation, return average RMSE 
def KFoldCV(k,X,y,model,flag):
    kf=KFold(n_splits=k)
    RMSE=[]
    for train,test in kf.split(X):
        model.fit(X[train],y[train])
        estimated=model.predict(X[test,:])
        RMSE.append(rmse(estimated,y[test],flag))
    return np.mean(RMSE)

# function for calculate root mean squared error 
def rmse(estimated,target,option):
    if option=='MSE':
#		return np.mean((estimated-target)**2)
		return metrics.mean_squared_error(estimated,target)
    else:
        return np.sqrt(((estimated-target)**2).mean())

#use dictionary to store alphas and corresponding minimum MSE
dic_lasso={}
dic_ridge={}

for alpha in alphas:
	lasso=Lasso(alpha=alpha)
	ridge=Ridge(alpha=alpha)
	MSE_lasso=KFoldCV(5,X_train,y_train,lasso,'MSE')
	MSE_ridge=KFoldCV(5,X_train,y_train,ridge,'MSE')
	dic_lasso[alpha]=MSE_lasso
	dic_ridge[alpha]=MSE_ridge
best_alpha_lasso=min(dic_lasso,key=dic_lasso.get)
best_alpha_ridge=min(dic_ridge,key=dic_ridge.get)
print '\nUsing Lasso,the best alpha is: ',best_alpha_lasso,' the min mse is: ',dic_lasso[best_alpha_lasso]
print '\nUsing Ridge,the best alpha is: ',best_alpha_ridge,' the min mse is: ',dic_ridge[best_alpha_ridge]

coefs_lasso=np.empty([X.shape[1],len(alphas)])
coefs_ridge=np.empty([X.shape[1],len(alphas)])
for i in range(len(alphas)):
    alpha=alphas[i]
    lasso=Lasso(alpha=alpha)
    ridge=Ridge(alpha=alpha)
    lasso.fit(X_train,y_train)
    ridge.fit(X_train,y_train)
    coefs_lasso[:,i]=lasso.coef_
    coefs_ridge[:,i]=ridge.coef_  
# plot 
import matplotlib.pyplot as plt

plt.figure()
alphas=alphas[::-1]
plt.plot(alphas,coefs_lasso[0,:])
for i in range(coefs_lasso.shape[0]):
    plt.plot(alphas,coefs_lasso[i,:])
plt.title('lasso')
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.figure()
for i in range(coefs_lasso.shape[0]):
    plt.plot(alphas,coefs_ridge[i,:])
plt.title('ridge')
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.yticks([1,-1])
plt.figure()
plt.plot(alphas,coefs_lasso[0,:],marker='*',color='red')
plt.show()
#2
class SGD:
    coef_=np.empty([4,1])
    def fit(X,y,r,max_iter):
        n=X.shape[0]
        coef=np.ones(4)
        for i in range(max_iter):
            for j in range(n):
                coef[0]=coef[0]-r*2*(coef[0]+coef[1]*X[j,0]+coef[2]*X[j,0]*X[j,1]+coef[3]*X[j,1])
                coef[1]=coef[1]-r*2*(coef[0]+coef[1]*X[j,0]+coef[2]*X[j,0]*X[j,1]+coef[3]*X[j,1])*X[j,0]
                coef[2]=coef[2]-r*2*(coef[0]+coef[1]*X[j,0]+coef[2]*X[j,0]*X[j,1]+coef[3]*X[j,1])*X[j,0]*X[j,1]
                coef[3]=coef[3]-r*2*(coef[0]+coef[1]*X[j,0]+coef[2]*X[j,0]*X[j,1]+coef[3]*X[j,1])*X[j,1]
        this.coef_=coef
    def predict(X):
        return coef_[0]+coef_[1]*X[:,0]+coef_[2]*np.dot(X[:,0],X[:,1])+coef_[3]*X[:,1]
