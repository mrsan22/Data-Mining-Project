__author__ = 'sanjiv'

import os, sys, datetime, csv, glob
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

n = 10 # number of folds for cv
sites = ['Dominant-Wrist','Dominant-Hip','Dominant-Thigh','Dominant-Ankle','Dominant-Upper-Arm']
activities = ['walking:-natural','cycling:-70-rpm_-50-watts_-.7-kg','sitting:-legs-straight','lying:-on-back']

def loso(x):
    return [[el for el in x if el!=x[i]] for i in range(len(x))]

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(activities))
    plt.xticks(tick_marks, activities, rotation=45)
    plt.yticks(tick_marks, activities)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    count = 0

    for site in sites:
        subList = glob.glob('train_subjects_33/*-features.csv')
        l = loso(subList)
        cm_comb = np.empty((len(subList),4,4))
        cm_comb[:] = np.NAN

        selector = SelectKBest(f_classif, k = 10)
        count = 0
        sub = 0
        for subgroup in l:
            # test data
            testfileObj = pd.read_csv(subList[sub])
            # print(site,(subList[sub]).split('/')[-1].split('-')[0])
            sub = sub + 1
            test_site_fileObj = (testfileObj.loc[testfileObj['SensorLocation'] == site])

            y_test = test_site_fileObj[['Activity']].values.ravel()
            X_test = test_site_fileObj[['MeanSM','StDevSM','MdnSM', 'belowPer25SM','belowPer75SM', 'TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values

            if X_test.shape[0] != 0:

                First=True
                for file in subgroup:
                    # train data
                    fileObj = pd.read_csv(file)
                    site_fileObj = (fileObj.loc[fileObj['SensorLocation'] == site])
                    y_cur = site_fileObj[['Activity']].values.ravel()
                    X_cur = site_fileObj[['MeanSM','StDevSM','MdnSM', 'belowPer25SM','belowPer75SM', 'TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
                    if First:
                        X_train = X_cur
                        y_train = y_cur
                        First = False
                    else:
                        X_train = np.concatenate((X_train,X_cur),axis=0)
                        y_train = np.concatenate((y_train,y_cur),axis=0)



                # feature selection
                X_train_red = selector.fit_transform(X_train,y_train)
                X_test_red = selector.transform(X_test)
                knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='uniform')
                param_grid = {
                    'n_neighbors': range(3,11,2),
                    'algorithm': ['auto', 'ball_tree','kd_tree'],
                    'weights': ['uniform','distance'],
                }
                knn_gs = GridSearchCV(knn, param_grid=param_grid)
                knn_gs.fit(X_train_red, y_train)
                y_pred = knn_gs.predict(X_test_red)

                cm = confusion_matrix(y_test, y_pred, labels = activities)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_comb[count,:,:] = cm_normalized
                count = count + 1


        # Compute the arithmetic mean along the specified axis, ignoring NaNs
        cm_mn = np.nanmean(cm_comb,axis=0)
        print(site,cm_mn[0,0],cm_mn[1,1],cm_mn[2,2],cm_mn[3,3])

        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(cm_mn, title='LOSO '+ site + ' Normalized confusion matrix')
        plt.savefig('LOSO_'+ site + '_confusion_matrix.png')
        plt.show()

main()
