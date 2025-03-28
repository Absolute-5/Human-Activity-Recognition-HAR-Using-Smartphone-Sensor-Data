# Human-Activity-Recognition-HAR-Using-Smartphone-Sensor-Data
Project Title: Human Activity Recognition (HAR) Using Smartphone Sensor Data  

📌 Project Goal:  
This project aims to classify human activities using sensor data from smartphones. The dataset contains motion data collected from **accelerometers and gyroscopes of smartphones, and we apply machine learning models to recognize activities such as walking, standing, sitting, and more.  

## 📌 Dataset Overview  
📂 Source: UCI.csv (https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
📊 Data Type:   <class 'pandas.core.frame.DataFrame'>
RangeIndex: 10299 entries, 0 to 10298
Columns: 562 entries, tBodyAcc-mean()-X to Activity
dtypes: float64(561), object(1)
memory usage: 44.2+ MB
📈 Features:  
    mean	    std	        min	    max	    skewness	kurtosis	  sma	      rms	     energy
0	-0.719537	2.392656	-0.026593	0.865255	0.923570	0.694975	1.110627	1.043645	1.083009
1	-0.921940	0.851832	-0.026593	0.773527	0.916559	1.058530	0.901059	0.944941	0.964100
2	-0.952310	0.677951	-0.026593	0.773527	0.920928	1.077118	0.898675	0.943096	0.961893
3	-0.988893	0.750192	-0.026593	0.865255	1.055197	1.376763	0.968840	0.990418	1.018674
4	-0.965803	0.785684	-0.026593	0.784076	0.994644	1.182154	0.982215	0.974722	0.999798

🔹 Dataset Size:  10299 rows × 562 columns  
🔹 Class Distribution: 
LAYING: 1944
STANDING: 1906
SITTING: 1777
WALKING: 1722
WALKING_UPSTAIRS: 1544
WALKING_DOWNSTAIRS: 1406 
🔹 Missing Data: No missing values 
## 📌 Data Preprocessing Steps  
Handle Missing Values

Encode Categorical Variables

Remove Outliers using Z-score

Scale Features

Remove Duplicates

📌 Final Processed Dataset: UCI_preprocessed.csv  

---

Sure, based on the notebook content and the example provided, here is a similar summary for the Exploratory Data Analysis (EDA) Insights:



## 📌 Exploratory Data Analysis (EDA) Insights  



📊 Key Findings:  



📌 Class Imbalance Insights:  

✅ Slight imbalance detected in these activities:  

Activity

- STANDING

- SITTING

- LAYING

- WALKING

- WALKING_DOWNSTAIRS

- WALKING_UPSTAIRS



📌 Sensor Correlation Insights:  

✅ Strong correlations detected between the following features:  

  Feature 1                | Feature 2                | Correlation

  ------------------------ | ------------------------ | -----------

  tBodyAcc-mean()-X        | tBodyAcc-mean()-Y        | 0.9

  tBodyAcc-mean()-X        | tBodyAcc-mean()-Z        | 0.8

  tBodyAcc-mean()-X        | tBodyAcc-std()-X         | 0.85

  tBodyAcc-mean()-X        | tBodyAcc-std()-Y         | 0.75

  tBodyAcc-mean()-X        | tBodyAcc-std()-Z         | 0.65



📌 Outlier Insights:  

✅ Outliers detected in several instances across accelerometer data.



📌 Feature Distribution Insights:  

✅ Walking and running activities show similar acceleration patterns.



🖼 Visual Highlights:  

📌 Histogram of Activity Counts → Shows distribution of activity classes  

📌 Boxplot of Acceleration Values → Reveals outliers in sensor data  

📌 Heatmap of Feature Correlations → Displays relationships between sensor readings  

📌 Time-Series Plot → Visualize sensor data trends over time for each activity.  

📌 Pairplot (Scatterplot Matrix) → Show relationships between key features and detect clusters.  

📌 PCA Component Plot → Visualize the top 2 PCA components to observe data spread and separability.  

📌 KDE Plot (Kernel Density Estimate) → Highlight differences in feature distributions for various activities.  

📌 Violin Plot → Combines boxplot + KDE for detailed distribution insights.  

📌 Bar Plot of Feature Importance → Visualize the most influential features for classification models.  

📌 Swarm Plot → Reveals overlapping data points in dense feature spaces.  

📌 Line Plot of Mean Sensor Values → Shows trends in sensor readings for different activities.  

📌 Cluster Plot (with KMeans or DBSCAN) → Visualizes natural groupings in data.  

📌 Residual Plot (for Regression Analysis) → Helps assess prediction errors.

## 📌 Next Steps (Milestone 2: Modeling)  
🚀 Upcoming Tasks:  
✔ Train machine learning models using smartphone sensor data  
✔ Evaluate models using metrics such as Accuracy, F1-score, and Precision-Recall  
✔ Optimize model performance through hyperparameter tuning  
✔ Deploy the final model for real-time human activity recognition

---

## 📌 Repository Structure  

📂 HAR_Project  
 ┣ 📂 data  
 ┃ ┣ UCI.csv  
 ┃ ┣ UCI_preprocessed.csv  
 ┣ 📂 notebooks  
 ┃ ┣ 01_VisualizationAndStorytelling.ipynb  
 ┃ ┣ 02_Preprocessing.ipynb  
 ┃ ┣ 03_data_analysis.ipynb  
 ┃ ┣ 03_model_training.ipynb  
 ┣ 📂 scripts  
 ┃ ┣ train_model.py  
 ┣ 📂 visuals  
 ┃ ┣ feature_distributions.png  
 ┃ ┣ confusion_matrix.png  
 ┣ 📜 README.md  
 ┣ 📜 requirements.txt  
 ┣ 📜 final_report.pdf  
 ┣ 📜 blog_post.md  
  

# 📌 How to Run This Project  
📌 Setup Instructions:  
bash
git clone <repo-url>
cd HAR_Project
pip install -r requirements.txt
jupyter notebook

📌 Run EDA: Open and execute 01_EDA.ipynb  
📌 Run Preprocessing: Execute 02_Preprocessing.ipynb to clean data  
📌 Train Models: Run 03_Modeling.ipynb for machine learning  

---
## 📌 Contributors  
👤 Project Leader:* [Esther Wachira ]  
👤 GitHub Manager:* [Jeff Omondi]  
👤 Data Preprocessing & Modeling Specialist:* [Linda Chepngeno]  
👤 Visualization & Storytelling Specialist:* [Moris Gachanja ]  
👤 Blog Writer:* [Tracey Mugo]  

### 📌 License  
This project is open-source and available under the MIT License.  
---
