# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:25:17 2021

@author: wolffbs
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix


class MetabolomicsML:
    ''' 
    This requires data file inputs:
        metabolomicsFile has metabolite measurement values
        metaboliteFile has information about the metabolites
        patientFile has other information from the patients (demographics, behavior, etc.)
    '''

    def __init__(self,
                 metabolomicsFile='data/metabolomicsData.csv',
                 metaboliteFile='data/metaboliteInfo.csv',
                 patientFile='data/patientData.csv'
                 ):
        self.mData = pd.read_csv(metabolomicsFile,index_col=0)
        self.mInfo = pd.read_csv(metaboliteFile,index_col=0)
        self.pData = pd.read_csv(patientFile,index_col=0)
        self.pDataCat = pd.DataFrame() # categorical versions of pData
        self.labels = pd.Series('') # labels for classifier
        self.components = pd.DataFrame() # principle components
        self.mDataPCs = np.empty(0) # data transformed by principle components
        self.feature_importance_avg = np.empty(0) # mean feature importance
        
    def preprocess(self,displayPlots=False):
        # make metabolites (features) columns
        self.mData = self.mData.T
        self.mData.index = self.mData.index.astype(int)
        
        # some metabolites are obviously unhelpful because all values are the same
        validMetabolites = self.mData.std() > 0.001
        self.mData = self.mData.loc[:,validMetabolites]
        self.mInfo = self.mInfo.loc[validMetabolites]
        
        # concentration measurements are ordinarily log-transformed
        self.mData = self.mData.applymap(np.log)
        
        # histogram of everything, looks like data are fairly normal, 
        # but there are definitely some outliers
        if displayPlots:
            self.mData.stack().hist(bins=200)       

    def removeOutliers(self,zThresh=3,replace_with='median'):
        # This function replaces any data with a z-score that is zThresh standard
        # deviations from the mean. These values are replaced with values
        # determined by "replace_with" ("mean", "median", or "mode")
        lim0 = self.mData.mean() - zThresh * self.mData.std()
        lim1 = self.mData.mean() + zThresh * self.mData.std()
        replace_values = getattr(self.mData,replace_with)()

        num_replaced = 0
        for c in self.mData.columns:
            to_replace = ~self.mData[c].between(lim0[c],lim1[c])
            num_replaced += to_replace.sum()
            self.mData.loc[to_replace,c] = replace_values[c]
        replaced = 100 * num_replaced / self.mData.size
        print('replaced {0:.2f}% of values with {1}'.format(replaced,replace_with))
        
    def scale(self,scaler=StandardScaler):
        # Scale data using standard scaling by default.
        # With this metabolomics data, MinMaxScaler produces very similar results
        self.mData = pd.DataFrame(scaler().fit_transform(self.mData),
                                  index=self.mData.index,
                                  columns=self.mData.columns)
    
    def pca(self,n_components=7):
        # Principle components analysis
        model = PCA(n_components=n_components)
        self.mDataPCs = pd.DataFrame(model.fit_transform(self.mData),index=self.mData.index)
        self.components = pd.DataFrame({i: c for i,c in enumerate(model.components_)},
                                       index=self.mData.columns)
        var = 100 * model.explained_variance_ratio_.sum()
        print('PCA: {0} components account for {1:.2f}% of variance'.format(n_components,var))
        
    def createCategorical(self,**kwargs):
        # creates categorical variables from continuous variables for classifying
        # has defaults here but can override with kwargs
        # can also input e.g. "mean" or "median" to split data based on that
        cutoffs = {'FACT-F': 41, # based on established clinical standards
                   'PROMIS': 50, # based on established clinical standards
                   'HAM-D': 2,   # based on established clinical standards
                   'BMI': 'mean'} # arbitrary
        for key,val in kwargs.items():
            cutoffs[key] = val

        for key,val in cutoffs.items():
            if isinstance(val,str):
                self.pDataCat[key] = self.pData[key] >= getattr(self.pData[key],val)()
            else:
                self.pDataCat[key] = self.pData[key] >= val    
            
    def classifier(self,labelName,modelType=SVC,displayPlots=True,**kwargs):
        # use SVM or random forest to classify data. kwargs go into model

        # set and store model name
        if modelType.__name__ == 'RandomForestClassifier':
            self.modelName = 'RFC'
        elif modelType.__name__ == 'SVC':
            self.modelName = 'SVM'
        else:
            assert False, 'wrong modelType, use SVC or RandomForestClassifier'
        
        # set labels
        try:
            self.labels = self.pDataCat[labelName]
        except KeyError:
            self.labels = self.pData[labelName]
        assert len(self.labels.unique()) < 6, '{} is probably not categorical'.format(labelName)
        self.labels = self.labels.dropna()
        
        # use PCA data if it exists, otherwise use data
        if self.mDataPCs.size:
            data = self.mDataPCs
        else:
            data = self.mData
            
        # create numpy arrays of only the data that has labels
        data = data.loc[self.labels.index].to_numpy()
        labelsArray = self.labels.to_numpy()
        
        # establish train, test sets
        skf = StratifiedKFold(n_splits=5)
        
        # create classifier model
        model = modelType(**kwargs)           

        # run the data through the classifier
        predicted,actual = [],[] # predicted class, actual class
        accuracy = [] # fraction of predictions that are correct
        feature_importance = [] 
        for train_index, test_index in skf.split(data, labelsArray):
            feat_train, label_train = data[train_index], labelsArray[train_index] # train data
            feat_test, label_test = data[test_index], labelsArray[test_index] # test data
            
            model.fit(feat_train,label_train)
            predictions = model.predict(feat_test)
            if self.modelName == 'RFC':
                feature_importance.append(model.feature_importances_)
            elif self.modelName == 'SVM':
                feature_importance.append(model.coef_)
           
            accuracy.append(accuracy_score(predictions,label_test))
            predicted.extend(predictions)
            actual.extend(label_test)
            
        print('Accuracy total = {0:.1f}%'.format(100 * np.mean(accuracy)))
        self.feature_importance_avg = np.mean(np.row_stack(np.abs(feature_importance)),
                                              axis=0)
        
        if displayPlots:
            title = '{}: {}'.format(modelType.__name__,labelName)
            def plotConfusionMatrix(act,pred,title=None):
                cm = confusion_matrix(act,pred)
                fig,ax = plt.subplots()
                sns.heatmap(pd.DataFrame(cm),annot=True,fmt='d',annot_kws={'fontsize':14},ax=ax)
                ax.set_xlabel('Predicted',fontsize=14)
                ax.set_ylabel('Actual',fontsize=14)
                if title is not None:
                    ax.set_title(title)
                fig.tight_layout()
            plotConfusionMatrix(actual,predicted,title)
            
            def plotImportPCA(featImport,title=None):
                fig,ax = plt.subplots()
                ax.bar(range(len(featImport)),featImport)
                ax.set_ylabel('Importance',fontsize=14)
                ax.set_xlabel('Principle Component',fontsize=14)
                if title is not None:
                    ax.set_title(title)
                fig.tight_layout()
            plotImportPCA(self.feature_importance_avg,title)
            
    def setMetaboliteImportance(self,write=False):
        # add "Importance" column to metaboliteInfo and sort descending
        # "Importance" is the product of:
        #     1. the importance assigned to each principle component by the ML model
        #     2. the PCA coefficients for each metabolite
        mImportance = self.components.abs() * self.feature_importance_avg
        mImportAvg = mImportance.mean(axis=1).rename('Importance')
        mImportAvg /= 100 * mImportAvg.sum() # change to percent
        self.mInfo = pd.concat((mImportAvg,self.mInfo),axis=1)
        self.mInfo.sort_values('Importance',ascending=False,inplace=True)
        
        # optionally write metabolite importance as csv in "results" folder 
        if write:
            fSuffix = '_'.join((self.modelName,self.labels.name))
            fName = 'results/metaboliteInfo_{}.csv'.format(fSuffix)
            if not os.path.isdir('results'):
                os.mkdir('results')
            self.mInfo.to_csv(fName)
            print('written to: {}'.format(fName))
        

#%% run linear SVM

def linearSVM(label='Healthy',write=False,zThresh=4,n_components=7):
    svm = MetabolomicsML()
    svm.preprocess()
    svm.removeOutliers(zThresh=zThresh)
    svm.scale()
    svm.pca(n_components=n_components)
    svm.createCategorical()
    svm.classifier(label,displayPlots=True,kernel='linear')
    svm.setMetaboliteImportance(write)

if __name__ == '__main__':
    labelNames = ['Healthy','FACT-F','BMI']
    write = [True,True,False] # don't bother writing BMI
    
    print('\nLINEAR SUPPORT VECTOR MACHINE')
    for labelName,w in zip(labelNames,write):
        print('\n\t{}'.format(labelName))
        linearSVM(labelName,w) 

#%% run random forest classifier

def randomForest(label='Healthy',write=False,zThresh=3,n_components=7,**kwargs):
    # kwargs get passed into RandomForestClassifier()
    rfc = MetabolomicsML()
    rfc.preprocess()
    rfc.removeOutliers(zThresh=zThresh)
    rfc.scale()
    rfc.pca(n_components=n_components)
    rfc.createCategorical()
    rfc.classifier(label,modelType=RandomForestClassifier,**kwargs)
    rfc.setMetaboliteImportance(write)
    
if __name__ == '__main__':
    labelNames = ['Healthy','FACT-F','BMI']
    write = [True,True,False] # don't bother writing BMI
    
    print('\nRANDOM FOREST CLASSIFIER')
    for labelName,w in zip(labelNames,write):
        print('\n\t{}'.format(labelName))
        randomForest(labelName,w,n_estimators=60,min_samples_split=5)

