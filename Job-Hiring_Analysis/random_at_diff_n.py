import csv
import numpy as np
import pandas as pd
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

index_for_iloc=3
for elements in list_of_files :

	index_for_iloc += 1
	index = []
	feature_list=[]
	ig_list=[]
	num_fields = 0
	mean = 0.0


	string_1 = elements[:-4]
	string_1 = string_1+'_test_constant_data.csv'
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

	X_train = pd.read_csv('p_train.csv')
	y_train = pd.read_csv('t_train.csv')
	X_test = pd.read_csv('p_test.csv')
	y_test = pd.read_csv('t_test.csv')

	############## Removing Columns I don't want ###############
	############################################################

	X_train = X_train.drop(X_train.columns[0], axis=1)
	X_test = X_test.drop(X_test.columns[0], axis=1)
	y_train = y_train.iloc[0:,index_for_iloc]
	y_test= y_test.iloc[0:,index_for_iloc]
	n_estim = 250
	while ( n_estim < 1001) :
		fp.write('{0},'.format(n_estim))
		for count in range(1,11) :
			
			########### Using Random Forest Regression ##################
			################## to the dataset ###########################

			from sklearn.ensemble import RandomForestRegressor
			regressor = RandomForestRegressor(n_estimators = n_estim, random_state = 0)
			regressor.fit(X_train, y_train)


			y_pred = regressor.predict(X_test)
			# print(regressor.score(X_train,y_train))
			# print(regressor.score(X_test,y_test))
			mean += regressor.score(X_test,y_test)
		mean /= 10
		fp.write('{:.5f}'.format(mean))
		fp.write('\n')
		n_estim += 25
	fp.close()