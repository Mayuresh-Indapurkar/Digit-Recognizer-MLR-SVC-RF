# AUTHOR - MAYURESH INDAPURKAR (MAX)
# WRITTEN IN PYTHON VERSION 3.5 USING SPYDER 2.3.9 IDE
# DIGIT CLASSIFIER USING PCA, RANDOM FOREST, SVC AND MULTINOMIAL LOGISTIC REGRESSION

#-----------------------------IMPORT REQUIRED LIBRARIES-----------------------------
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#-----------------------------IMPORT REQUIRED LIBRARIES-----------------------------


if __name__ == '__main__':    
    
    #-----------------------------DATA IMPORT STARTS-----------------------------
    train = pd.read_csv('M:/DataScience/Masters/3.AppliedMultivariateAnalysis/HW2-DigitRecognizer/data/train.csv')
    test = pd.read_csv('M:/DataScience/Masters/3.AppliedMultivariateAnalysis/HW2-DigitRecognizer/data/test.csv')
    #-----------------------------DATA IMPORT ENDS-----------------------------
    
    
    #-----------------------------EXPLORATORY ANALYSIS STARTS-----------------------------
    label = train[[0]].values.ravel() #SEGREGATE THE TARGET VARIABLE AND TRAIN DATA
    traindata = train.ix[:,1:].values #SEGREGATE THE TARGET VARIABLE AND TRAIN DATA
    pca = PCA() #PRINCIPAL COMPONENT ANALYSIS
    pca.fit(traindata)	 #FIT THE PCA ON TRAIN DATA
    var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)    
    plt.plot(var) #FIRST 50 COMPONENTS ACCOUNT FOR ~80% VARIANCE
    pca = PCA(n_components=50,whiten=True) #PCA WITH FIRST 50 COMPONENTS  
    traindata_pca = pca.transform(traindata)	#ORTHOGONAL TRANSFORMATION OF TRAIN DATA
    test_pca = pca.transform(test) #ORTHOGONAL TRANSFORMATION OF TEST DATA 	
    #-----------------------------EXPLORATORY ANALYSIS ENDS-----------------------------    
    
    
    #-----------------------------MODEL #1 RANDOM FOREST STARTS-----------------------------
    rf=RandomForestClassifier(n_estimators=50) 
    params = {'n_estimators': [500,1000]}
    grf = GridSearchCV(rf, params, refit='True', n_jobs=-1, cv=3) #HYPERPARAMETER TUNING USING GRID-SEARCH-CV
    grf.fit(traindata_pca,label) #FIT RANDOM FOREST ON TRAIN DATA
    pred = grf.predict(test_pca) #PREDICT USING TRAINED MODEL
    print("Best Params for Random Forest: " + str(grf.best_params_)) #PRINT THE BEST PERFORMING PARAMETERS FROM GRID
    np.savetxt('M:/DataScience/Masters/3.AppliedMultivariateAnalysis/HW2-DigitRecognizer/testOP/RandomForest.csv',np.c_[range(1,len(test)+1),pred],delimiter=',',header='ImageId,Label',comments='',fmt='%d')
    #-----------------------------MODEL #1 RANDOM FOREST ENDS-----------------------------


    #-----------------------------MODEL #2 SUPPORT VECTOR CLASSIFICATION STARTS-----------------------------	
    svc = SVC()
    params = {'C':[100,500], 'tol': [0.0001]}
    gsvc = GridSearchCV(svc, params, refit='True', n_jobs=-1, cv=3) #HYPERPARAMETER TUNING USING GRID-SEARCH-CV
    gsvc.fit(traindata_pca,label) #FIT SVC ON TRAIN DATA
    pred = gsvc.predict(test_pca) #PREDICT USING TEST MODEL
    print("Best Params for SVC: " + str(gsvc.best_params_)) #THE BEST PERFORMING PARAMETERS
    np.savetxt('M:/DataScience/Masters/3.AppliedMultivariateAnalysis/HW2-DigitRecognizer/testOP/SVC.csv',np.c_[range(1,len(test)+1),pred],delimiter=',',header='ImageId,Label',comments='',fmt='%d')
    #-----------------------------MODEL #2 SUPPORT VECTOR CLASSIFICATION ENDS-----------------------------	


    #-----------------------------MODEL #3 MULTINOMIAL LOGISTIC REGRESSION STARTS-----------------------------	
    log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    params = {'C':[500,1000,1400], 'tol': [0.0001,0.00001]}
    mlr = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=-1, cv=3) #HYPERPARAMETER TUNING USING GRID-SEARCH-CV
    mlr.fit(traindata_pca,label)  #FIT SVC ON TRAIN DATA  
    pred = mlr.predict(test_pca) #PREDICT USING TEST MODEL
    print("Best Params for MLR: " + str(mlr.best_params_)) #THE BEST PERFORMING PARAMETERS
    np.savetxt('M:/DataScience/Masters/3.AppliedMultivariateAnalysis/HW2-DigitRecognizer/testOP/mlr.csv',np.c_[range(1,len(test)+1),pred],delimiter=',',header='ImageId,Label',comments='',fmt='%d')
    #-----------------------------MODEL #3 MULTINOMIAL LOGISTIC REGRESSION ENDS-----------------------------	
    
