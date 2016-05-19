import os, sys, datetime, csv, glob, itertools
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cross_validation import StratifiedKFold, train_test_split,KFold
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import NuSVC,SVC
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

n = 10 # number of folds for cv
sites = ['Dominant-Wrist','Dominant-Hip','Dominant-Thigh','Dominant-Ankle','Dominant-Upper-Arm']
activities = ['walking:-natural','cycling:-70-rpm_-50-watts_-.7-kg','sitting:-legs-straight','lying:-on-back']


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

count = 0


for site in sites:
    # print(site)
    subList = glob.glob('features/*-features.csv')
    cm_comb = np.empty((n,4,4))
    cm_comb[:] = np.NAN

    tmp = np.empty((n,3,4))
    tmp[:] = np.NAN

    selector = SelectKBest(f_classif, k = 10)
    count = 0
    First = True
    for sub in subList:
        fileObj = pd.read_csv(sub)
        test_site_fileObj = (fileObj.loc[fileObj['SensorLocation'] == site])
        y_cur = test_site_fileObj[['Activity']].values.ravel()
        X_cur = test_site_fileObj[['MeanSM','StDevSM','MdnSM', 'belowPer25SM','belowPer75SM', 'TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values

        if X_cur.shape[0] != 0:
            if First:
                X = X_cur
                y = y_cur
                First = False
            else:
                X = np.concatenate((X,X_cur),axis=0)
                y = np.concatenate((y,y_cur),axis=0)

    tenFold = KFold(len(y),n)
    for train_index, test_index in tenFold:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # feature selection
        X_train_red = selector.fit_transform(X_train,y_train)
        X_test_red = selector.transform(X_test)

        # rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion='gini',oob_score = False)
        rfc = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
        rfc.fit(X_train_red, y_train)
        y_pred = rfc.predict(X_test_red)

        cm = confusion_matrix(y_test, y_pred, labels = activities)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_comb[count,:,:] = cm_normalized

        pre_rec_fscore = precision_recall_fscore_support(y_test, y_pred, average=None,labels=activities)
        tmp[count,0,:],tmp[count,1,:],tmp[count,2,:] = pre_rec_fscore[0],pre_rec_fscore[1],pre_rec_fscore[2]
        # print(pre_rec_fscore[0],pre_rec_fscore[1],pre_rec_fscore[2])
        count = count + 1

    # Compute the arithmetic mean along the specified axis, ignoring NaNs

    tmp_mn = np.nanmean(tmp,axis=0)
    print(site,tmp_mn)
    # cm_mn = np.nanmean(cm_comb,axis=0)
    # print(site,cm_mn[0,0],cm_mn[1,1],cm_mn[2,2],cm_mn[3,3])

    # np.set_printoptions(precision=2)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.figure(figsize=(10,10))
    # plot_confusion_matrix(cm_mn, title='Ten-fold '+ site + ' Normalized confusion matrix')
    # plt.savefig('Ten_Fold_'+ site + '_confusion_matrix.png')
    # plt.show()

