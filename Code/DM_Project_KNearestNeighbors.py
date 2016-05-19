
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# # Using KNearestNeighbor for classification of activities:

# # Sensor located at Dominant Wrist:
# ## Reading the training files

# In[62]:

dominant_wrist_file = pd.read_csv('sensor_based_files_train/dominant_wrist_train.csv')
train_data_wrist = dominant_wrist_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_wrist = dominant_wrist_file[['Activity']].values.ravel()


# In[63]:
print "Training data dimensions for sensor at dominant_wrist position"
print train_data_wrist.shape
print target_label_wrist.shape
print "----------------------------------------------------------------------------"
print "\n"

# ## Reading the testing files

# In[64]:

dominant_wrist_file_test = pd.read_csv('sensor_based_files_test/dominant_wrist_test.csv')
test_data_wrist = dominant_wrist_file_test[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_wrist_test = dominant_wrist_file_test[['Activity']].values.ravel()


# In[65]:
print "Testing data dimensions for sensor at dominant_wrist position"
print test_data_wrist.shape
print target_label_wrist_test.shape
print "----------------------------------------------------------------------------"
print "\n"

# ## Using K-NearestNeighbors

# In[66]:

knn_wrist = KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='uniform')
param_grid = {
    'n_neighbors': range(3,11,2),
    'algorithm': ['auto', 'ball_tree','kd_tree'],
    'weights': ['uniform','distance'],
}
knn_wrist_gs = GridSearchCV(knn_wrist, param_grid=param_grid)
knn_wrist_gs.fit(train_data_wrist, target_label_wrist)
predicted = knn_wrist_gs.predict(test_data_wrist)
print "Best parameters to be used for training the model",knn_wrist_gs.best_params_
print "\n"

# In[67]:
print "Classification report for sensor at Ankle position"
# Evaluation
print metrics.classification_report(target_label_wrist_test, predicted)
print metrics.confusion_matrix(target_label_wrist_test, predicted)
print "F-Score: ",metrics.f1_score(target_label_wrist_test, predicted)


# In[68]:

activities = ['sitting:-legs-straight','cycling:-70-rpm_-50-watts_-.7-kg','walking:-natural','lying:-on-back']
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


# Compute confusion matrix
cm = confusion_matrix(target_label_wrist_test, predicted, labels=activities)
np.set_printoptions(precision=2)
# print('Confusion matrix, without normalization')
# print(cm)
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix for Sensor at Dominant-Wrist position')
plt.savefig('confusion_matrix.png')
plt.show()


# ## 10Fold Cross Validation

# In[69]:

# Reading all 33 subject files
dominant_wrist_file = pd.read_csv('sensor_based_files_33/dominant_wrist_train.csv')
train_data_wrist = dominant_wrist_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_wrist = dominant_wrist_file[['Activity']].values.ravel()


# In[70]:

# print train_data_wrist.shape
# print target_label_wrist.shape


# In[17]:
print "Beginning 10-Fold cross validation on all the 33 subjects for sensor at wrist position..."
clf_cross_val = KNeighborsClassifier(n_neighbors=9, algorithm='auto', weights='uniform')
scores = cross_val_score(clf_cross_val, train_data_wrist,target_label_wrist,cv=10)
print "Scores for each fold:"
print scores
print "---------------------------------------------------------------------------------"
print ("Accuracy for 10Fold cross validation using KNN: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print "\n"

# # Sensor located at Upper-Arm
# ## Reading Training files

# In[71]:

dominant_upperarm_file = pd.read_csv('sensor_based_files_train/dominant_Upper_Arm_train.csv')
train_data_upperarm = dominant_upperarm_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_upperarm = dominant_upperarm_file[['Activity']].values.ravel()


# In[72]:
print "Training data dimensions for sensor at dominant_upper_arm position"
print train_data_upperarm.shape
print target_label_upperarm.shape
print "----------------------------------------------------------------------------"
print "\n"

# ## Reading Testing files

# In[73]:

dominant_upperarm_file_test = pd.read_csv('sensor_based_files_test/dominant_Upper_Arm_test.csv')
test_data_upperarm = dominant_upperarm_file_test[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_upperarm_test = dominant_upperarm_file_test[['Activity']].values.ravel()


# In[74]:
print "Testing data dimensions for sensor at dominant_upper_arm position"
print test_data_upperarm.shape
print target_label_upperarm_test.shape
print "----------------------------------------------------------------------------"
print "\n"

# ## Using K_NearestNeighbors

# In[75]:

knn_upperarm = KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='uniform')
param_grid = {
    'n_neighbors': range(3,11,2),
    'algorithm': ['auto', 'ball_tree','kd_tree'],
    'weights': ['uniform','distance'],
}
knn_arm_gs = GridSearchCV(knn_upperarm, param_grid=param_grid)
knn_arm_gs.fit(train_data_upperarm, target_label_upperarm)
predicted = knn_arm_gs.predict(test_data_upperarm)
print "Best parameters to be used for training the model",knn_arm_gs.best_params_
print "\n"

# In[76]:
print "Classification report for sensor at Upper Arm position"

# Evaluation
print metrics.classification_report(target_label_upperarm_test, predicted)
print metrics.confusion_matrix(target_label_upperarm_test, predicted)
print "F1-Score",metrics.f1_score(target_label_upperarm_test, predicted)


# In[77]:

activities = ['sitting:-legs-straight','cycling:-70-rpm_-50-watts_-.7-kg','walking:-natural','lying:-on-back']
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


# Compute confusion matrix
cm = confusion_matrix(target_label_upperarm_test, predicted, labels=activities)
np.set_printoptions(precision=2)
# print('Confusion matrix, without normalization')
# print(cm)
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix for Sensor at Domiant-upper_arm position')
plt.savefig('confusion_matrix.png')
plt.show()
print "--------------------------------------------------------------------------------------"
print "\n"


# ## 10Fold Cross Validation

# In[27]:

# Reading all 33 subject files
dominant_upperarm_file = pd.read_csv('sensor_based_files_33/dominant_Upper_Arm_train.csv')
train_data_upperarm = dominant_upperarm_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_upperarm = dominant_upperarm_file[['Activity']].values.ravel()


# In[28]:
#
# print train_data_upperarm.shape
# print target_label_upperarm.shape


# In[29]:
print "Beginning 10-Fold cross validation on all the 33 subjects for sensor at UpperArm position..."
clf_cross_val = KNeighborsClassifier(n_neighbors=9, algorithm='auto', weights='distance')
scores = cross_val_score(clf_cross_val, train_data_upperarm,target_label_upperarm,cv=10)
print "Scores for each fold:"
print scores
print "---------------------------------------------------------------------------------"
print ("Accuracy for 10Fold cross validation using KNN: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# # Sensor located at Thigh
# ## Reading Training Files

# In[78]:

dominant_thigh_file = pd.read_csv('sensor_based_files_train/dominant_Thigh_train.csv')
train_data_thigh = dominant_thigh_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_thigh = dominant_thigh_file[['Activity']].values.ravel()


# In[79]:
print "Training data dimensions for sensor at dominant_thigh position"
print train_data_thigh.shape
print target_label_thigh.shape
print "----------------------------------------------------------------------------"
print "\n"

# ## Reading Testing files

# In[80]:

dominant_thigh_file_test = pd.read_csv('sensor_based_files_test/dominant_Thigh_test.csv')
test_data_thigh = dominant_thigh_file_test[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_thigh_test = dominant_thigh_file_test[['Activity']].values.ravel()


# In[81]:
print "Testing data dimensions for sensor at dominant_thigh position"
print test_data_thigh.shape
print target_label_thigh_test.shape
print "----------------------------------------------------------------------------"
print "\n"


# ## Using K_NearestNeighbors

# In[82]:

knn_thigh = KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='uniform')
param_grid = {
    'n_neighbors': range(3,11,2),
    'algorithm': ['auto', 'ball_tree','kd_tree'],
    'weights': ['uniform','distance'],
}
knn_thigh_gs = GridSearchCV(knn_thigh, param_grid=param_grid)
knn_thigh_gs.fit(train_data_thigh, target_label_thigh)
predicted = knn_thigh_gs.predict(test_data_thigh)
print "Best parameters to be used for training the model",knn_thigh_gs.best_params_
print "\n"

# In[83]:

# Evaluation
print "Classification report for sensor at Thigh position"
print metrics.classification_report(target_label_thigh_test, predicted)
print metrics.confusion_matrix(target_label_thigh_test, predicted)
print "F1-Score",metrics.f1_score(target_label_thigh_test, predicted)


# In[84]:

activities = ['sitting:-legs-straight','cycling:-70-rpm_-50-watts_-.7-kg','walking:-natural','lying:-on-back']
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


# Compute confusion matrix
cm = confusion_matrix(target_label_thigh_test, predicted, labels=activities)
np.set_printoptions(precision=2)
# print('Confusion matrix, without normalization')
# print(cm)
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix for Sensor at Dominant-Thigh position')
plt.savefig('confusion_matrix.png')
plt.show()
print "--------------------------------------------------------------------------------------"
print "\n"


# ## 10Fold Cross Validation

# In[39]:

# Reading all 33 subject files
dominant_thigh_file = pd.read_csv('sensor_based_files_33/dominant_Thigh_train.csv')
train_data_thigh = dominant_thigh_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_thigh = dominant_thigh_file[['Activity']].values.ravel()


# In[40]:


# In[41]:
print "Beginning 10-Fold cross validation on all the 33 subjects for sensor at Thigh position..."
clf_cross_val = KNeighborsClassifier(n_neighbors=9, algorithm='auto', weights='uniform')
scores = cross_val_score(clf_cross_val, train_data_thigh,target_label_thigh,cv=10)
print "Scores for each fold:"
print scores
print "---------------------------------------------------------------------------------"
print ("Accuracy for 10Fold cross validation using KNN: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# # Sensor Located at Hip
# ## Reading Training Files

# In[85]:

dominant_hip_file = pd.read_csv('sensor_based_files_train/dominant_Hip_train.csv')
train_data_hip = dominant_hip_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_hip = dominant_hip_file[['Activity']].values.ravel()


# In[86]:
print "Training data dimensions for sensor at dominant_hip position"
print train_data_hip.shape
print target_label_hip.shape
print "----------------------------------------------------------------------------"
print "\n"


# ## Reading Test Files

# In[87]:

dominant_hip_file_test = pd.read_csv('sensor_based_files_test/dominant_Hip_test.csv')
test_data_hip = dominant_hip_file_test[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_hip_test = dominant_hip_file_test[['Activity']].values.ravel()


# In[88]:
print "Testing data dimensions for sensor at dominant_hip position"
print test_data_hip.shape
print target_label_hip_test.shape
print "----------------------------------------------------------------------------"
print "\n"

# ## Using K-NearestNeighbor

# In[89]:

knn_hip = KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='uniform')
param_grid = {
    'n_neighbors': range(3,11,2),
    'algorithm': ['auto', 'ball_tree','kd_tree'],
    'weights': ['uniform','distance'],
}
knn_hip_gs = GridSearchCV(knn_hip, param_grid=param_grid)
knn_hip_gs.fit(train_data_hip, target_label_hip)
predicted = knn_hip_gs.predict(test_data_hip)
print "Best parameters to be used for training the model",knn_hip_gs.best_params_
print "\n"

# In[90]:

# Evaluation
print "Classification report for sensor at Hip position"
print metrics.classification_report(target_label_hip_test, predicted)
print metrics.confusion_matrix(target_label_hip_test, predicted)
print "F-Score: ",metrics.f1_score(target_label_hip_test, predicted)


# In[91]:

activities = ['sitting:-legs-straight','cycling:-70-rpm_-50-watts_-.7-kg','walking:-natural','lying:-on-back']
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


# Compute confusion matrix
cm = confusion_matrix(target_label_hip_test, predicted, labels=activities)
np.set_printoptions(precision=2)
# print('Confusion matrix, without normalization')
# print(cm)
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix for sensor located at Hip position')
plt.savefig('confusion_matrix.png')
plt.show()
print "--------------------------------------------------------------------------------------"
print "\n"

# ## 10Fold Cross validation

# In[49]:

# Reading all 33 subject files
dominant_hip_file = pd.read_csv('sensor_based_files_33/dominant_Hip_train.csv')
train_data_hip = dominant_hip_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_hip = dominant_hip_file[['Activity']].values.ravel()


# In[50]:

# print train_data_hip.shape
# print target_label_hip.shape
#

# In[51]:
print "Beginning 10-Fold cross validation on all the 33 subjects for sensor at Hip position..."
clf_cross_val = KNeighborsClassifier(n_neighbors=9, algorithm='auto', weights='uniform')
scores = cross_val_score(clf_cross_val, train_data_hip,target_label_hip,cv=10)
print "Scores for each fold:"
print scores
print "---------------------------------------------------------------------------------"
print ("Accuracy for 10Fold cross validation using KNN: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# # Sensor Located at Ankle
# ## Reading Training files

# In[92]:

dominant_ankle_file = pd.read_csv('sensor_based_files_train/dominant_Ankle_train.csv')
train_data_ankle = dominant_ankle_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_ankle = dominant_ankle_file[['Activity']].values.ravel()


# In[93]:
print "Training data dimensions for sensor at dominant_ankle position"
print train_data_ankle.shape
print target_label_ankle.shape
print "----------------------------------------------------------------------------"
print "\n"


# ## Reading Test files

# In[94]:

dominant_ankle_file_test = pd.read_csv('sensor_based_files_test/dominant_Ankle_test.csv')
test_data_ankle = dominant_ankle_file_test[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_ankle_test = dominant_ankle_file_test[['Activity']].values.ravel()


# In[95]:
print "Testing data dimensions for sensor at dominant_ankle position"
print test_data_ankle.shape
print target_label_ankle_test.shape
print "----------------------------------------------------------------------------"
print "\n"

# ## Using K-NearestNeighbors

# In[96]:

knn_ankle = KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='uniform')
param_grid = {
    'n_neighbors': range(3,11,2),
    'algorithm': ['auto', 'ball_tree','kd_tree'],
    'weights': ['uniform','distance'],
}
knn_ankle_gs = GridSearchCV(knn_ankle, param_grid=param_grid)
knn_ankle_gs.fit(train_data_ankle, target_label_ankle)
predicted = knn_ankle_gs.predict(test_data_ankle)
print "Best parameters to be used for training the model",knn_ankle_gs.best_params_


# In[97]:

# Evaluation
print "Classification report for sensor at Ankle position"
print metrics.classification_report(target_label_ankle_test, predicted)
print metrics.confusion_matrix(target_label_ankle_test, predicted)
print "F-Score: ",metrics.f1_score(target_label_ankle_test, predicted)


# In[98]:

activities = ['sitting:-legs-straight','cycling:-70-rpm_-50-watts_-.7-kg','walking:-natural','lying:-on-back']
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


# Compute confusion matrix
cm = confusion_matrix(target_label_ankle_test, predicted, labels=activities)
np.set_printoptions(precision=2)
# print('Confusion matrix, without normalization')
# print(cm)
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix for Sensor at Ankle position')
plt.savefig('confusion_matrix.png')
plt.show()
print "--------------------------------------------------------------------------------------"
print "\n"

# ## 10Fold Cross Validation

# In[59]:

# Reading all 33 subject files
dominant_ankle_file = pd.read_csv('sensor_based_files_33/dominant_Ankle_train.csv')
train_data_ankle = dominant_ankle_file[['MeanSM','StDevSM','MdnSM','belowPer25SM','belowPer75SM','TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15']].values
target_label_ankle = dominant_ankle_file[['Activity']].values.ravel()


# In[60]:

# print train_data_ankle.shape
# print target_label_ankle.shape


# In[61]:
print "Beginning 10-Fold cross validation on all the 33 subjects for sensor at Ankle position..."
clf_cross_val = KNeighborsClassifier(n_neighbors=9, algorithm='auto', weights='uniform')
scores = cross_val_score(clf_cross_val, train_data_ankle,target_label_ankle,cv=10)
print "Scores for each fold:"
print scores
print "---------------------------------------------------------------------------------"
print ("Accuracy for 10Fold cross validation using KNN: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

