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
# sites = ['Dominant-Hip','Dominant-Thigh','Dominant-Ankle']
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

def findsubsets(S,m):
    return (itertools.combinations(S, m))


def mergeDF(site_fileObj_1,site_fileObj_2,activities):
    columns=['Activity','MeanSM','StDevSM','MdnSM', 'belowPer25SM','belowPer75SM', 'TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15','MeanSM_s2','StDevSM_s2','MdnSM_s2', 'belowPer25SM_s2','belowPer75SM_s2', 'TotPower_0.3_15_s2','FirsDomFre_0.3_15_s2','PowFirsDomFre_0.3_15_s2','SecDomFre_0.3_15_s2','PowSecDomFre_0.3_15_s2','FirsDomFre_0.6_2.5_s2','PowFirsDomFre_0.6_2.5_s2','FirsDomFre_per_TotPower_0.3_15_s2']
    First = True
    for act in activities:
        toPick = min(site_fileObj_1.loc[site_fileObj_1['Activity']==act].shape[0],site_fileObj_2.loc[site_fileObj_2['Activity']==act].shape[0])
        if toPick != 0:
            new_site_1 = site_fileObj_1.loc[site_fileObj_1['Activity']==act][:toPick].drop('SensorLocation',1).as_matrix()
            new_site_2 = site_fileObj_2.loc[site_fileObj_2['Activity']==act][:toPick].drop(['SensorLocation','Activity'], 1).as_matrix()
            result_cur = np.concatenate((new_site_1,new_site_2),axis=1)
            if First:
                data4df = result_cur
                First = False
            else:
                data4df = np.concatenate((data4df,result_cur),axis=0)
    result = pd.DataFrame(data4df,columns=columns)
    return result


twoSites = findsubsets(sites,2)
count = 0

for site in twoSites:
    site = (list(site))
    siteStr = site[0] +'_'+ site[1]
    subList = glob.glob('features/*-features.csv')
    l = loso(subList)
    cm_comb = np.empty((len(subList),4,4))
    cm_comb[:] = np.NAN

    tmp = np.empty((len(subList),3,4))
    tmp[:] = np.NAN

    selector = SelectKBest(f_classif, k = 20)
    count = 0
    sub = 0
    for subgroup in l:
        # test data
        testfileObj = pd.read_csv(subList[sub])
        sub = sub + 1
        test_site_fileObj_1 = (testfileObj.loc[testfileObj['SensorLocation'] == site[0]])
        test_site_fileObj_2 = (testfileObj.loc[testfileObj['SensorLocation'] == site[1]])

        if((test_site_fileObj_1.shape[0] != 0)& (test_site_fileObj_2.shape[0] != 0)):
            test_finalMerged = mergeDF(test_site_fileObj_1,test_site_fileObj_2,activities)
            y_test = test_finalMerged[['Activity']].values.ravel()
            X_test = test_finalMerged[['MeanSM','StDevSM','MdnSM', 'belowPer25SM','belowPer75SM', 'TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15','MeanSM_s2','StDevSM_s2','MdnSM_s2', 'belowPer25SM_s2','belowPer75SM_s2', 'TotPower_0.3_15_s2','FirsDomFre_0.3_15_s2','PowFirsDomFre_0.3_15_s2','SecDomFre_0.3_15_s2','PowSecDomFre_0.3_15_s2','FirsDomFre_0.6_2.5_s2','PowFirsDomFre_0.6_2.5_s2','FirsDomFre_per_TotPower_0.3_15_s2']]

            First=True
            for file in subgroup:
                # train data
                fileObj = pd.read_csv(file)
                train_site_fileObj_1 = (fileObj.loc[fileObj['SensorLocation'] == site[0]])
                train_site_fileObj_2 = (fileObj.loc[fileObj['SensorLocation'] == site[1]])

                if((train_site_fileObj_1.shape[0] != 0)& (train_site_fileObj_2.shape[0] != 0)):
                    train_finalMerged = mergeDF(train_site_fileObj_1,train_site_fileObj_2,activities)
                    y_cur = train_finalMerged[['Activity']].values.ravel()
                    X_cur = train_finalMerged[['MeanSM','StDevSM','MdnSM', 'belowPer25SM','belowPer75SM', 'TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15','MeanSM_s2','StDevSM_s2','MdnSM_s2', 'belowPer25SM_s2','belowPer75SM_s2', 'TotPower_0.3_15_s2','FirsDomFre_0.3_15_s2','PowFirsDomFre_0.3_15_s2','SecDomFre_0.3_15_s2','PowSecDomFre_0.3_15_s2','FirsDomFre_0.6_2.5_s2','PowFirsDomFre_0.6_2.5_s2','FirsDomFre_per_TotPower_0.3_15_s2']]
                    if First:
                        X_train = X_cur
                        y_train = y_cur
                        First = False
                    else:
                        X_train = np.concatenate((X_train,X_cur),axis=0)
                        y_train = np.concatenate((y_train,y_cur),axis=0)

            # print(X_test.shape,y_test.shape, X_train.shape, y_train.shape)
            # print (siteStr,X_test)

            # feature selection
            X_train_red = selector.fit_transform(X_train,y_train)
            X_test_red = selector.transform(X_test)

            # rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion='gini',oob_score = False)
            #rfc = OneVsRestClassifier(SVC(kernel='rbf',C=100,gamma=0.1))
            # rfc = SVC(kernel='rbf', gamma=0.7,C=1,random_state=10)
            rfc = KNeighborsClassifier(n_neighbors=11,algorithm='auto')
            rfc.fit(X_train_red, y_train)
            y_pred = rfc.predict(X_test_red)

            cm = confusion_matrix(y_test, y_pred, labels = activities)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_comb[count,:,:] = cm_normalized

            pre_rec_fscore = precision_recall_fscore_support(y_test, y_pred, average=None,labels=activities)
            tmp[count,0,:],tmp[count,1,:],tmp[count,2,:] = pre_rec_fscore[0],pre_rec_fscore[1],pre_rec_fscore[2]
            count = count + 1


    # Compute the arithmetic mean along the specified axis, ignoring NaNs
    tmp_mn = np.nanmean(tmp,axis=0)
    print(siteStr)
    print(tmp_mn)
    cm_mn = np.nanmean(cm_comb,axis=0)
    print(cm_mn[0,0],cm_mn[1,1],cm_mn[2,2],cm_mn[3,3])
    #
    # np.set_printoptions(precision=2)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.figure(figsize=(10,10))
    # plot_confusion_matrix(cm_mn, title='LOSO '+ siteStr + ' Normalized confusion matrix')
    # plt.savefig('LOSO_'+ siteStr + '_confusion_matrix.png')
    # plt.show()



