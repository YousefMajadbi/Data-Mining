# CSE-572: Data Mining
## I.	INTRODUCTION
This repository contains three significant projects completed as part of the course CSE 572 “Data Mining”. These projects demonstrate the application of advanced data mining techniques like extracting time-series data, machine learning modeling, and data clustering. The projects highlight the practical skills and knowledge gained during the course.
The Extracting Time Series Properties of Glucose Levels in Artificial Pancreas Project provided hands-on experience with features extraction and data synchronization using Python libraries like Pandas and NumPy. The project involved implementing a Python code to load two datasets, synchronize the data, and calculate the hyperglycemia and hypoglycemia metrics from the data, which can help to determine the level of each metric for a given interval per day.
The Machine Model Training Project involved developing a code to train a machine model and calculate the accuracy of each model using scikit-learn, the model can be used to assess whether a person has eaten a meal or not eaten a meal.
And the Cluster Validation Project involved writing a program using Python that performs clustering on the dataset, and perform cluster validation to determine the amount of carbohydrates in each meal.

## II.	EXPLANATION OF THE SOLUTION
### A.	Project 1 – Load and Process Data
The project starts with loading two datasets, one from the Continuous Glucose Sensor (CGMData.csv) and the other from the Insulin pump (InsulinData.csv). And process the loaded data, first by extracting only necessary attributes like Timestamp and the 5-minute filtered CGM reading in mg/dL from CGM data, and Timestamp and auto mode exit events and unique codes representing reasons data from Insulin data. Second Segment CGM data where each segment corresponds to a day’s worth of data, and there should be 288 samples in each segment, then each segment divided into two sub-segments: the daytime sub-segment and the overnight sub-segment. Third, tackling missing data by interpolation. 

### B.	Project 1 – Data Synchronization
The second objective in this project is to synchronize the data from both of sensors, by determining the auto mode and manual mode of data insertion which is found in the “Auto Mode Active PLGM Off” column in the Insulin device data, then synchronize it with the CGM data by searching for the time stamp nearest to (and later than) the Auto mode start time stamp obtained from insulin data. 

### C.	Project 1 – Metrics Calculation
The third objective in this project is to compute and report overall statistical measures from data, by extracting the below metrics: 
1.	Percentage time in hyperglycemia (CGM > 180 mg/dL)
2.	Percentage of time in hyperglycemia critical (CGM > 250 mg/dL)
3.	Percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)
4.	Percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)
5.	Percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)
6.	Percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)
Where each of the above metrics are extracted in three different time intervals: daytime (6 am to midnight), overnight (midnight to 6 am), and whole day (12 am to 12 am). And computed for both manual and auto modes. 

### D.  Project 2 – Load and Process data
Started with loading the CGM and Insulin data for two patients and combined the datasets, then extracting meal and no meal data based on the [BWZ Carb Input (grams)] column where meal data comprises 2hr 30min stretch of CGM data, where each stretch is one row and have 30 columns. So eventually we created a meal matrix with (P x 30) where P is the total number of rows for the meal data time series, and no meal data comprises 2 hours stretch after the founded meal time and created a no meal matrix with (Q x 24) where Q is the total number of no meal data time series and 24 is the number of columns. And we handled missing data by using the interpolation method. 

### E.  Project 2 – Feature Extraction
An important step for the machine learning modeling is the feature extraction (or feature engineering), which was done the raw data of meal and no meal matrices which contain a lot of noise, so we extracted the following features: Mean, Variance, 4 elements of Fast Fourier Transform (FFT) and 2 levels of Difference Calculation. To end up with 8 extracted features.

### F.  Project 2 – Train Machine Learning models
We used the extracted features (after scaling them) to train our machine learning models, by using K-Fold cross-validation technique to split the data into training and testing data, fit the training data into the models to train them.

### G.  Project 3 – Clustering
In this project we clustered meal data based on the amount of carbohydrates in each meal which can be found in the [BWZ Carb Input (grams)] column from Insulin data. The clustering was done by discretizing the meal amount in bins of size 20, in total we had n = (max-min)/20 bins. And put each row from the meal data matrix that was generated in Project 2 in the respective bins according to their meal amount label. Then using the extracted featured from Project 2 to cluster the meal data into n clusters, using DBSCAN and KMeans.

### H.  Project 3 – Cluster Validation
To validate our clusters we computed SSE, Entropy and Purity values. For SSE the values were combined to get one Sum of Squared Error – SSE for both Kmeans and DBSCAN, with the below formula: 

SEE= ∑_(i=1)^K▒∑_(x∈C_i)▒〖dist^2 (m_i,x)〗

And for Entropy and Purity we created one matrices for each Kmeans and DBSCAN which contains bins as columns and clusters and rows, then calculated Entropy following this formula: 

Entropy(t)= -∑_j^t▒〖p(j│t)  log_2⁡〖p(j│t)〗 〗

And Purity following this formula: 

Purity=1/N ∑_(k=1)^K▒max⁡(n_(k,j) ) 

## III.	RESULTS
### A.	Project 1
The first project generated a csv file which includes 2×18 matrix, where the two rows for both Manual mode and Auto mode and the 18 columns contain the calculated metrics during all periods (overnight, daytime and whole day). ---- This demonstrated the power of data mining techniques to extract time series properties from large datasets. 
Findings: 
•	The patient has a critical or level 2 of hypoglycemia during the whole day. 
•	We can extract more calculated metrics from the data to forecast blood glucose levels, like "Bolus Volume Delivered," "Basal Rate," "Carb Input," and "Sensor Glucose".

### B.	Project 2
The second project produced an N×1 vector of Ones and Zeros, where if a row is determined to be a meal data, then the corresponding entry will be 1, and if determined to be no meal, the corresponding entry will be 0.
Findings: 
•	The Decision Tree model provided better performance than the SVM where the accuracy of Decision Tree was higher.
•	We can use this machine learning system to help the artificial pancreas medical control system to automatically determine where the patient eaten a meal or not eaten a meal, as currently it requires the patient to add this information manually.

### C.	Project 3
The third project output file contains a 1×6 vector, with the following format [“SSE for KMeans”, “SSE for DBSCAN”, “Entropy for KMeans”, “Entropy for DBSCAN”, “Purity for KMeans”, “Purity for DBSCAN”]. 
Findings: 
•	We can use the clustering project in determining whether the eaten meal is high or low in carbohydrates.
•	KMeans might be preferred for its simplicity and effectiveness in forming clear clusters
•	DBSCAN could be more useful in scenarios where noise and outliers are significant, and the data does not conform to spherical cluster shapes.

## IV.	CONTRIBUTIONS
### A.	Project 1 – Extracting Time Series Properties
In the Extracting Time Series Properties of Glucose
Levels in Artificial Pancreas Project, I contributed by extracting key performance metrics from time series data captured by the Medtronic 670G Artificial Pancreas system, which includes Continuous Glucose Monitor (CGM) and insulin pump data. My contributions focused on processing and synchronizing data from the two sensors, handling missing data, and calculating glucose metrics under different operational modes.
Task 1 – Data Processing: This task included extracting relevant columns for both loaded datasets, segmenting the CGM data into daily intervals, and dividing it into daytime and overnight sub-segments. I also addressed missing data through interpolation techniques, ensuring the data was clean and complete for accurate metric calculations.
Task 2 – Data Synchronization: I implemented a function to synchronize time stamps between the CGM and insulin pump datasets, which operate asynchronously. This involved identifying the auto-mode start time in the insulin pump data and aligning it with the corresponding time in the CGM data.
Task 3 – Feature Extraction: I extracted multiple glucose metrics, including percentage time spent in different glycemic ranges (hyperglycemia, hypoglycemia, and in-range) during the day, overnight, and over a 24-hour period. For each day, I calculated the mean values of these metrics across manual and auto modes
My contributions involved utilizing Python and libraries like pandas and numpy to preprocess and analyze large time series datasets, ensuring synchronization, segmentation, and feature extraction were completed with high accuracy.

### B.	Machine Model Training Project
In this project, I developed and trained a machine learning model to distinguish between meal and no meal time series data from Continuous Glucose Monitor (CGM) and insulin pump datasets. The goal was to predict whether a person had eaten a meal based on glucose data, using supervised classification techniques.
Task 1 – Data Extraction: I extracted meal and no meal data from both CGM and insulin datasets. I located meal times by identifying non-zero values in the insulin dataset’s carb input column, then extracted corresponding glucose values from the CGM dataset. Where each meal comprises 2-hour 30-minute stretch of glucose readings, with a 30-minute period before the meal and a 2-hour postprandial period. For no meal data, I extracted 2-hour stretches from the post-absorptive period where no meals occurred. I handled multiple conditions in data extraction, such as overlapping meals and stretches with missing data.
Task 2 – Data Processing: I processed the extracted data by creating a Meal Data Matrix (P x 30) and a No Meal Data Matrix (Q x 24), where P and Q represent the number of samples. Missing data was handled using interpolation strategy. 
Task 3 – Feature Extraction: I implemented a feature extraction function to extract necessary features like Mean, Variance, 4 elements of Fast Fourier Transform (FFT) and 2 levels of Difference Calculation. The generated feature matrices which contained 8 columns were scaled using the StandaredScaler of the Python library sklearn.preprocessing.
Task 4 – Machine learning model training: I trained a machine learning model using supervised classification, where I concatenated the meal and no meal feature matrices and labeled the meal data as 1 and the no meal data as 0. I used scikit-learn’s SVM and DecisionTreeClassifier algorithms to train the model. Additionally, I used  k-fold cross-validation to evaluate model performance, ensuring the model’s robustness and accuracy. The models then used to predict outcomes for new test data and classify it as either meal or no meal.
I used Python and libraries such as scikit-learn, pandas, numpy, and scipy to extract features, train the model, and evaluate performance, successfully completing the task of meal detection from glucose time series data. 

### C.	Project 3 – Cluster Validation Project
In this project, I implemented a cluster solution to analyze and validate clusters of meal data based on the amount of carbohydrates consumed. Where the goal was to apply clustering techniques to Continuous Glucose Monitor (CGM) and Insulin data, then evaluate the clustering accuracy using Sum of Squared Errors (SSE), Entropy, and Purity metrics. 
Task 1 – Ground Truth Extraction: The first step involved extracting the ground truth labels from the InsulinData.csv file. I took the BWZ Carb Input (grams) column, which represents the carbohydrate intake during each meal. I then discretized the carbohydrate values into bins of size 20, ensuring that each bin represented a range of 20 grams of carbohydrates. These bins were used to assign each meal data sample from the Meal Data Matrix (P x 30) created in the previous project to its respective ground truth bin.
Task 2 – Feature Extraction: I reused the feature extraction function that was implemented in the previous project to extract necessary features like Mean, Variance, 4 elements of Fast Fourier Transform (FFT) and 2 levels of Difference Calculation. The generated feature matrices which contained 8 columns were scaled using the StandaredScaler of the Python library sklearn.preprocessing.
Task 3 – Clustering with KMeans and DBSCAN: I implemented two clustering algorithms—KMeans and DBSCAN—to cluster the meal data based on the extracted features. Both algorithms grouped the meals into n clusters, where n represented the number of carbohydrate bins derived from the ground truth. KMeans clustered the data by minimizing the distance between points and cluster centroids, while DBSCAN grouped data points based on density.
Task 4 – Clusters Validation: After clustering the data, I calculated the SSE, Entropy, and Purity for both KMeans and DBSCAN. Where the entropy and purity computed by constructing two matrices for each algorithm, the columns of the matrices represented the carbohydrate bins (b1, b2, …, bn), and the rows represented the clusters (C1, C2, …, Cn). I populated the matrices by mapping the cluster values to their respective bins. Using these matrices, I calculated Entropy, which measures the degree of disorder within the clusters, and Purity, which measures the extent to which each cluster contains only members of a single class.
I used Python and key libraries such as scikit-learn, pandas, numpy, scipy, and matplotlib to perform the clustering, compute validation metrics, and generate the required outputs.

## V.	LESSONS LEARNED
### A.	Extracting Time Series Properties Project
Through this project, I learned the following: 
1.	How to process time series data effectively, particularly in the context of healthcare systems like continuous glucose monitoring (CGM).
2.	The challenges involved synchronizing data from two asynchronously operating devices (CGM sensor and insulin pump) and how to overcome them. 
3.	Advanced techniques for handling missing data, such as interpolation. 
4.	Leveraging Python libraries like Pandas and NumPy to manage and analyze large time series datasets effectively. 

### B.	Machine Model Training Project 
Through this project, I learned the following: 
1.	The process of extracting relevant features from time series data to distinguish between different classes (meal and no meal). 
2.	Advanced techniques for handling missing data, such as interpolation.
3.	How to implement and train machine learning models using scikit-learn, focusing on supervised classification methods like SVM and Decision Trees.
4.	The importance of k-fold cross-validation for assessing model performance and generalization.

### C.	Cluster Validation Project 
Through this project, I learned the following: 
1.	How to perform clustering on time series data to identify patterns and group similar data points based on specific features. Using different clustering algorithms such as KMeans and DBSCAN. 
2.	Techniques for calculating clustering performance metrics, such as SSE, Entropy, and Purity, to evaluate the effectiveness of clustering results.
3.	Utilizing Python libraries like scikit-learn and Pandas for clustering and data manipulation tasks effectively.

## VI.	REFERENCES
[1]	Supervised Machine Learning: A Brief Primer. Available at: Supervised Machine Learning: A Brief Primer (sciencedirectassets.com)
[2]	Support vector machine: A tool for mapping mineral prospectivity. Available at: Support vector machine A tool for mapping mineral prospectivity (sciencedirectassets.com) 
[3]	Application of the DBSCAN Algorithm to Detect Hydrophobic Clusters in Protein Structures. Available at: Cryst1903017Lashkov.fm (asu.edu)
[4]	Automatic Extraction and Cluster Analysis of Natural Disaster Metadata Based on the Unified Metadata Framework. Available at Automatic Extraction and Cluster Analysis of Natural Disaster Metadata Based on the Unified Metadata Framework (mdpi-res.com)
[5]	Scipy Interpolation: Interpolation (scipy.interpolate) — SciPy v1.14.1 Manual
[6]	Scipy Fourier Transforms: Fourier Transforms (scipy.fft) — SciPy v1.14.1 Manual 
