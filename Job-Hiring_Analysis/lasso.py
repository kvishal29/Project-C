import csv
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def bubbleSort(arr,arr2,arr3):
    n = len(arr)
    for i in range(1,n):
        for j in range(1, n-i-1):
            if arr[j] < arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
                arr2[j], arr2[j+1] = arr2[j+1], arr2[j]
                arr3[j], arr3[j+1] = arr3[j+1], arr3[j]

####################### Variables ###########################
#############################################################

list_of_files = ['4_Colleague.csv','5_Engaged.csv','6_Excited.csv','7_EyeContact.csv','8_Smiled.csv','9_SpeakingRate.csv','10_NoFillers.csv','11_Friendly.csv','12_Paused.csv','13_EngagingTone.csv','14_StructuredAnswers.csv','15_Calm.csv','16_NotStressed.csv','17_Focused.csv','18_Authentic.csv','19_NotAwkward.csv']

maxx=0
meann= 0.0
mean_last = 0
index_for_iloc=4

for elements in list_of_files :
	index = []
	feature_list=[]
	ig_list=[]
	num_fields = 0
	mean = 0.0

	string_1 = elements[:-4]
	string_1 = string_1+'_svm.csv'
	fp = open(string_1,'w')

	############## Extracting the required fields ###############
	#############################################################

	with open(elements) as csvfile :
		csvreader = csv.reader(csvfile)

		for i in range(0,5):
			fields = csvreader.__next__()
			if i == 0 :
				feature_list = fields
				num_fields = len(fields)
			if i == 4 :
				ig_list = fields

	#### Making index list of same length as Feature list#######
	############################################################

	for i in range(0,num_fields) :
		index.append(i) 

	######### Sorting the lists in descending order ############
	######### for extracting the top features when required ####
	############################################################

	bubbleSort(ig_list,feature_list,index)

	####################  Dataset ##############################
	############################################################

	X = pd.read_csv('prosody_modified.csv')
	y = pd.read_csv('turker_scores.csv')

	############## Removing Columns I don't want ###############
	############################################################

	X = X.drop(X.columns[0], axis=1)
	for i in range(8,len(index)):
		X = X.drop(feature_list[i], axis=1)
	
	# fp.write('{0},'.format(n_estim))
	for count in range(1,2) :
		X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1)
		y_train = y_train.iloc[0:,index_for_iloc]
		y_test= y_test.iloc[0:,index_for_iloc]

		from sklearn.linear_model import Lasso
		
		#Fit the model
		lasso = linear_model.Lasso()
		lasso.set_params(alpha=5.0)
		lasso.fit(X_train, y_train)

		from sklearn.model_selection import cross_val_score
		accuracies = cross_val_score(estimator = lasso, X = X_train, y = y_train, cv = 5)
		for elem in accuracies:
			fp.write('{:.5f}'.format(elem))
			fp.write('\n')
		fp.write('{:.5f}'.format(accuracies.mean()))
		fp.write('\n')
		fp.write('{:.5f}'.format(accuracies.std()))
		# mean += lasso.score(X_test, y_test)
		meann = accuracies.mean()
		# print(meann)
		# print(accuracies.std())

		#SVR algorithm for training purpose
		# from sklearn.svm import SVR
		# regressor = SVR(kernel='rbf')
		# regressor.fit(x_tr,y_tr)

	# mean /= 10
	# fp.write('{:.5f}'.format(mean))
	# fp.write('\n')
	index_for_iloc+=1
	if mean_last < meann :
		mean_last=meann
print(mean_last)
