import csv, sys
import datetime
import os
import pandas as pd


def push_to_file(csv_list):
	if len(csv_list) >= 5000:
		with open(subjectFile, 'a') as out:
			output = csv.writer(out, delimiter=',')
			output.writerows(csv_list)
			csv_list = []
	return csv_list

data_folder = "Stanford2010Data"
data_file_list = ["RawCorrectedData_Dominant-Wrist.csv", "RawCorrectedData_Dominant-Hip.csv",
				  "RawCorrectedData_Dominant-Thigh.csv", "RawCorrectedData_Dominant-Upper-Arm.csv",
				  "RawCorrectedData_Dominant-Ankle.csv"]

dirs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
#output_file = "print-merged.csv"
csvList = []
for folder in dirs:
	try:
		subject = folder
		subjectFile = "output/" + subject + "-activity.csv"
		print("Subject: " + subjectFile)
		with open(subjectFile, 'w') as out:
			output = csv.writer(out, delimiter=',')
			output.writerow(["X", "Y", "Z", "SensorLocation" ,"Timestamp", "Activity"])
		folder = data_folder + "\\" + folder + "\\" + "merged"
		annotation_file = folder + "\\" + "AnnotationIntervals.csv"
		print("Annotation: " + annotation_file)
		timeLabelList = []
		with open(annotation_file, 'r') as f:
			reader = csv.reader(f)
			isFirst = True
			for row in reader:
				if len(row) != 5:
					continue
				if isFirst:
					isFirst = False
					continue
				timeLabelList.append([row[0], row[1], row[3], row[4]])

		activityDict = {}

		csvList = push_to_file(csvList)
		for data_file in os.listdir(folder):
			if not data_file.startswith("Wocket_"):
				continue
			wocket_file = folder + "\\" + data_file
			if os.path.isdir(wocket_file):
				continue
			expectedName = False
			for name in data_file_list:
				if(data_file.__contains__(name)):
					expectedName = True
					break

			if not expectedName:
				continue

			activity_arr = data_file.split("_")
			activity_name = activity_arr[len(activity_arr) - 1].split(".")[0]

			print("Wocket: " + wocket_file)
			wocket_file_names = ["TimeStamp", "X", "Y", "Z"]
			wocket_df = pd.read_csv(wocket_file, names=wocket_file_names)

			wocket_df["TimeStamp"] = wocket_df.apply(lambda w_row:
													 	datetime.datetime.utcfromtimestamp(w_row['TimeStamp'] / 1000.0),
													 axis=1)

			for timeLabel in timeLabelList:
				try:
					csvList = push_to_file(csvList)
					label = ""
					if(timeLabel[3] == "walking:-natural" or timeLabel[3] == "cycling:-70-rpm_-50-watts_-.7-kg"):
						label = timeLabel[3]
					elif(timeLabel[2] == "lying:-on-back" or timeLabel[2] == "sitting:-legs-straight"):
						label = timeLabel[2]
					else:
						continue

					start_date = datetime.datetime.strptime(timeLabel[0], "%Y-%m-%d %H:%M:%S")
					end_date = datetime.datetime.strptime(timeLabel[1], "%Y-%m-%d %H:%M:%S")
					start_date = start_date + datetime.timedelta(seconds=2)
					end_date = end_date - datetime.timedelta(seconds=2)

					start_wocket_entry = (wocket_df.loc[wocket_df['TimeStamp'] == start_date])
					end_wocket_entry = (wocket_df.loc[wocket_df['TimeStamp'] == end_date])

					if len(start_wocket_entry.index.tolist()) < 1 or len(end_wocket_entry.index.tolist()) < 1:
					   continue;

					start_index = start_wocket_entry.index.tolist()[0]
					end_index = end_wocket_entry.index.tolist()[0]

					req_data_from_wocket = wocket_df.ix[start_index : end_index, :]

					epoch = datetime.datetime.utcfromtimestamp(0)
					for row in req_data_from_wocket.iterrows():
						time_stamp = (row[1]["TimeStamp"] - epoch).total_seconds() * 1000.0
						csvList.append([row[1]["X"], row[1]["Y"], row[1]["Z"], activity_name, time_stamp, label])
				except Exception as ex:
					tb = sys.exc_info()[2]
					print(ex.with_traceback(tb))
	except Exception as ex:
		tb = sys.exc_info()[2]
		print(ex.with_traceback(tb))
	csvList = push_to_file(csvList)
	print(folder + " Processed")
