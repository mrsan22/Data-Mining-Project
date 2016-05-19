# Data-Mining
Recognizing human activity using multiple wearable accelerometer sensors placed at different body positions.

This project was created as a part of Data Mining course at Northeastern University.

In this project, we implemented and evaluated classification algorithm to detect four crucial human physical activities (walking, cycling, sitting, and lying) using five triaxial accelerometers worn concurrently on different parts of the body (dominant hip, upper arm, ankle, thigh, and wrist). The accelerometer data were collected, cleaned, and preprocessed to extract features from 10 s window.

 These time and frequency domain features were used with Random Forest and k-Nearest Neighbour classifier to classify subject activities. The algorithms were evaluated based on Leave-One-Subject-Out (LOSO) and ten-fold cross-validation strategy using both accelerometer data as well as annotated activity labels from 33 participants in a lab. 
 
Random Forest showed the best performance recognizing the activities with overall accuracy of 89 % for LOSO strategy for hip data. Combining data from both hip and ankle improved the overall accuracy by 3.5 %, and by 10% for lying activity, which had the lowest classification accuracy (80%) for hip data.

* Technologies Used:
* Language: Python
* Tools: IPython, PyCharm
* Libraries: Scikit-Learn/SciPy, NumPy, Pandas, Matplotlib
* Algorithms : Random Forest, K-NN
* Evaluation Methodology: LOSO(Leave One Subject Out), K-Fold validation, Confusion Matrix
