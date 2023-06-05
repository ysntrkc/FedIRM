import os, shutil
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_PERCENTAGE = 0.8
datasets = ['isic2019', 'ham10000', 'rsna']

if os.path.exists(datasets[0]):
	os.rename(os.path.join(datasets[0], 'ISIC_2019_Training_Input', 'ISIC_2019_Training_Input'), os.path.join(datasets[0], 'images'))

	# remove the 'ISIC_2019_Training_Input' folder and ISIC_2019_Training_Metadata.csv file
	shutil.rmtree(os.path.join(datasets[0], 'ISIC_2019_Training_Input'))
	os.remove(os.path.join(datasets[0], 'ISIC_2019_Training_Metadata.csv'))

	# read the 'ISIC_2019_Training_GroundTruth.csv' file
	isic_data = pd.read_csv(os.path.join(datasets[0], 'ISIC_2019_Training_GroundTruth.csv'))
	isic_data = isic_data.rename(columns={'image': 'ImageID'})
	isic_data['ImageID'] = isic_data['ImageID'] + '.jpg'
	isic_data.drop(['UNK'], axis=1, inplace=True)

	# obtain class labels
	classes = isic_data.columns[1:]

	# change column data types float to int
	isic_data[classes] = isic_data[classes].astype(int)

	# create empty dataframe to store train and test data
	isic_train = pd.DataFrame(columns=isic_data.columns)
	isic_test = pd.DataFrame(columns=isic_data.columns)

 	# split the data into train and test sets
	for c in classes:
		class_data = isic_data[isic_data[c] == 1]
		train, test = train_test_split(class_data, train_size=TRAIN_PERCENTAGE, random_state=42)
		isic_train = pd.concat([isic_train, train])
		isic_test = pd.concat([isic_test, test])

  # create train and test csv files
	isic_train.to_csv(os.path.join(datasets[0], 'train.csv'), index=False)
	isic_test.to_csv(os.path.join(datasets[0], 'test.csv'), index=False)

	# remove the 'ISIC_2019_Training_GroundTruth.csv' file
	os.remove(os.path.join(datasets[0], 'ISIC_2019_Training_GroundTruth.csv'))