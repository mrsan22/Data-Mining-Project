All the steps of data mining are completely done using python
Dependencies:
		pandas
		numpy
		scipy
		sklearn
		matplotlib
		pylab

	- Install dependencies, steps below:
		Open cmd
		Goto C:\Python27\Scripts (In windows)
		Run "pip install pandas"
		Run "pip install numpy"
		Run "pip install scipy"
		Run "pip install sklearn"
		Run "pip install pylab"

You can find the required data at the below url:
https://www.dropbox.com/sh/i3fgtw3zzr4r0ih/AAApsAIat0tF_xo1locMD8ara

You can find a view only copy of the code and the output at the following urls:
http://nbviewer.ipython.org/gist/mrsan22/90519bd3be6f2f8cd33b
The above url shows following code/output for Random Forest:
	- Random Forest algorithm
	- 10-fold cross validation using Random Forest
	- Leave-One-Subject-Out cross validation using Random Forest

http://nbviewer.ipython.org/gist/mrsan22/52e5f4c3fd96baa74d32	
The above url shows following code/output kNN:
	- kNN algorithm
	- 10-fold cross validation using kNN
	- Leave-One-Subject-Out cross validation using kNN
		
preprocessor.py
	This is the preprocessing step. It combines the Annotation file having the time interval and the Wocket 
	accelerometer file data.
	Running the program:
		- Have the data folder "Stanford2010Data" at the same location as the preprocessor.py file
			Your directory should look like -> "Stanford2010Data\dec0209\merged\Wocket_*.csv"
		- Create a folder "output" at the same location
		- Run the file using "python preprocessor.py"
		
getFeatures.py
	This is the feature extraction step. It extracts all the features mentioned in the project report.
		Running the program:
		- Have the data folder "output" at the same location as the getFeatures.py file. This is the output 
		from the preprocessing step.
			Your directory should look like -> "output\*-activity.csv"
		- Create a folder "features" at the same location
		- Run the file using "python getFeatures.py"

DM_Project_KNearestNeighbors.py
	This is the classification step which uses kNN for classification. It also runs 10-fold cross validation on the data.
		Running the program:
		- Have the data folder "sensor_based_files_train" at the same location as the DM_Project_KNearestNeighbors.py file
			Your directory should look like -> "sensor_based_files_train\*.csv"	
		- Have the data folder "sensor_based_files_test" at the same location as the DM_Project_KNearestNeighbors.py file
			Your directory should look like -> "sensor_based_files_test\*.csv"	
		- Have the data folder "sensor_based_files_33" at the same location as the DM_Project_KNearestNeighbors.py file
			Your directory should look like -> "sensor_based_files_33\*.csv"	
		- Run the file using "python DM_Project_KNearestNeighbors.py"
		
DM_Project_RandomForest.py
	This is the classification step which uses Random Forest for classification. It also runs 10-fold cross validation on the data.
		Running the program:
		- Have the data folder "sensor_based_files_train" at the same location as the DM_Project_RandomForest.py file
			Your directory should look like -> "sensor_based_files_train\*.csv"	
		- Have the data folder "sensor_based_files_test" at the same location as the DM_Project_RandomForest.py file
			Your directory should look like -> "sensor_based_files_test\*.csv"	
		- Have the data folder "sensor_based_files_33" at the same location as the DM_Project_RandomForest.py file
			Your directory should look like -> "sensor_based_files_33\*.csv"	
		- Run the file using "python DM_Project_RandomForest.py"

LOSO_KNN.py
	This is the LOSO (Leave-One-Subject-Out) validation step using kNN
		Running the program:
		- Have the data folder "sensor_based_files_33" at the same location as the LOSO_KNN.py file
			Your directory should look like -> "sensor_based_files_33\*.csv"	
		- Have the data folder "train_subjects_33" at the same location as the LOSO_KNN.py file
			Your directory should look like -> "train_subjects_33\*.csv"
		- Run the file using "python LOSO_KNN.py"

LOSO_RandomForest.py
	This is the LOSO (Leave-One-Subject-Out) validation step using Random Forest
		Running the program:
		- Have the data folder "sensor_based_files_33" at the same location as the LOSO_RandomForest.py file
			Your directory should look like -> "sensor_based_files_33\*.csv"	
		- Have the data folder "train_subjects_33" at the same location as the LOSO_RandomForest.py file
			Your directory should look like -> "train_subjects_33\*.csv"
		- Run the file using "python LOSO_RandomForest.py"
		
classifyMultipleSites.py
	This program evaluates the performance of each pair of sensor placement sites
		Running the program:
		- Have the data folder "features" at the same location as the classifyMultipleSites.py file
			Your directory should look like -> "features\*.csv"	
		- Run the file using "python classifyMultipleSites.py"
		
classify_10fold.py
	This program runs 10-fold cross validation on the data using kNN and RandomForest
		- Have the data folder "features" at the same location as the classifyMultipleSites.py file
			Your directory should look like -> "features\*.csv"	
		- For kNN:
			- Comment lines 114, 115, 116
			- Uncomment line 117
		- For Random Forest:
			- Comment lines 115, 116, 117
			- Uncomment line 114
		- Run the file using "python classify_10fold.py"