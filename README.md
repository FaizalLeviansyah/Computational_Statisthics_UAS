Computational_Statisthics_UAS
Computational statistics is a branch of statistics that focuses on the development and use of computer algorithms to solve statistical problems. It is a rapidly growing field, as the increasing availability of computational power has made it possible to analyze large and complex datasets. we get the material from our lecturer named : Mr.Septian Enggar

==============================
<Background>
==============================
  Over time, the medicine’s effective time shortens, causing discomfort among PD patients. 
Thus, PD patients and clinicians must monitor and record the patient symptom changes for adequate treatment. 
Parkinson's disease (PD) is a slowly progressive nervous system condition caused by the loss of dopamine-producing brain cells. 
It primarily affects the patient's motor abilities, but it also has an impact on non-motor functions over time. 
Patients' symptoms include tremors, muscle stiffness, and difficulty walking and balancing. 
Then it disrupts the patients' sleep, speech, and mental functions, affecting their quality of life (QoL). With the improvement of wearable devices, patients can be continuously monitored with the help of cell phones and fitness trackers. We use these technologies to monitor patients' health, record WO periods, and document.
the impact of treatment on their symptoms in order to predict the wearing-off phenomenon in Parkinson's disease patients.

==============================
<Task>
==============================
Using multiple linear regression, try to find the best regression model that is suitable to represent the case: 
helping the doctors to create specific treatment strategies to manage Parkinson's disease and its associated symptoms properly. 
It means you are asked to create a model that can anticipate the "wearing-off" of anti-Parkinson Disease medication.
==============================

(StatKom_UAS_.ipynb)
==============================
Description 
==============================

1. <import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
import csv
from google.colab import auth
from google.colab import files
from  sklearn import linear_model>
 = above function is to import the library

2.<drive.mount('/content/drive')>
= The drive.mount() function in Google Colab is used to mount your Google Drive into the Colab runtime environment. This allows you to immediately access files and directories in your Google Drive from your Colab notebook. The '/content/drive' parameter gives the directory path where you wish to mount your Google Drive. In this scenario, it mounts the root directory of your Google Drive to the Colab runtime's /content/drive path. When you run this code, you will be prompted to approve access to your Google Drive account. After successful authorisation, you can access your Google Drive files and folders in your Colab notebook by using the /content/drive path.

3. import gdown

# Define the Google Sheets URL
spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1_hsbboROP0BGGcddHl6JmU5b-7iaool7Cz3Glgp9OPA/edit?usp=sharing'
= The import gdown sentence loads the gdown library, which enables file downloads from the internet.The spreadsheet_url variable contains the URL of the Google Sheets document to be downloaded.
# Get the file ID from the URL
file_id = spreadsheet_url.split('/')[-2]
= The spreadsheet_url variable contains the URL of the Google Sheets document to be downloaded.
# Define the export URL for CSV format
export_url = f'https://docs.google.com/spreadsheets/export?id={file_id}&exportFormat=csv'
= The file_id variable parses the URL to obtain the file ID. The file ID is the document's unique identification in Google Sheets. The export_url variable is created by combining the file ID with the CSV export URL pattern. It specifies the URL for downloading the Google Sheets document in CSV format.
# Download the CSV file
output_file = '/content/combined_data.csv'  # Specify the path to save the file
gdown.download(export_url, output_file, quiet=False)
= The output_file variable stores the location of the downloaded CSV file on the local workstation


4. <data_df = pd.read_csv('combined_data.csv')
   data_df>
= the code reads data from a CSV file called 'combined_data.csv' and creates a pandas DataFrame named data_df. The DataFrame is then displayed on the screen.

5. ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/d551c506-cdd5-427f-8097-307e01d745e8)

= The above code appears to be a snippet of a dataset containing various health-related measurements for a participant. Here's a breakdown of the different components:

- timestamp: It represents the date and time at which the measurements were recorded.
- heart_rate: The heart rate of the participant at the given timestamp.
- steps: The number of steps taken by the participant.
- stress_score: A measure of the participant's stress level.
- awake, deep, light, rem, nonrem_total: These columns likely represent different sleep stages (awake, deep, light, REM, non-REM) and provide information about the participant's sleep patterns.
- total: The total duration of sleep.
- nonrem_percentage: The percentage of non-REM sleep.
- sleep_efficiency: A measure of how efficiently the participant slept.
- time_from_last_drug_taken: The time elapsed since the participant took their last drug.
- wo_duration: The duration of a workout session.
- timestamp_hour, timestamp_dayofweek, timestamp_hour_sin, timestamp_hour_cos: These columns may represent derived features from the timestamp, such as the hour of the day, the day of the week, and their corresponding sine and cosine values.
- wearing_off: It could be a binary indicator representing the wearing off of a particular drug.
- participant: Identifies the participant to whom the measurements belong.
-The code snippet shows a subset of the dataset, displaying 672 rows and 20 columns. Each row represents a specific timestamped measurement for the participant, capturing various health-related parameters over time.

6.  data_df.isnull().sum()
    dfnew = data_df.dropna()
   x = dfnew[['time_from_last_drug_taken', 'timestamp_hour', 'timestamp_dayofweek']]
   y = dfnew['wearing_off']
   y

= the code checks for missing values in the DataFrame data_df, drops any rows with missing values to create a new DataFrame dfnew, and then selects specific columns as x and the 'wearing_off' column as y. The resulting x DataFrame will contain the columns 'time_from_last_drug_taken', 'timestamp_hour', and 'timestamp_dayofweek', while y will contain the 'wearing_off' column.

7. from sklearn.preprocessing import MinMaxScaler

# Inisialisasi objek MinMaxScaler
scaler = MinMaxScaler()

# Melakukan normalisasi data
normalized_data = scaler.fit_transform(x)

# Menampilkan hasil normalisasi
print(normalized_data)

Output :
: [[0.28571429 0.         0.66666667]
 [0.30952381 0.         0.66666667]
 [0.33333333 0.         0.66666667]
 ...
 [0.10634921 1.         0.33333333]
 [0.13015873 1.         0.33333333]
 [0.15396825 1.         0.33333333]]
 
= The code is to demonstrates the usage of the `MinMaxScaler` class from the `sklearn.preprocessing` module in scikit-learn. This class is commonly used for feature scaling or normalization of numerical data.
In the code, the `MinMaxScaler` object is initialized with the following line:

- scaler = MinMaxScaler()
This creates an instance of the `MinMaxScaler` class.

The next step is to normalize the data using the `fit_transform` method:
<python>
normalized_data = scaler.fit_transform(x)

Here, `x` represents the input data that you want to normalize. The `fit_transform` method computes the minimum and maximum values of each feature in `x` and then performs the normalization calculation to transform the data. The resulting normalized data is stored in the `normalized_data` variable.
Finally, the normalized data is printed with the following line:

- print(normalized_data)
This code simply displays the normalized data on the console.

The output displayed is a numpy array representing the normalized values of the features in `x`. Each row corresponds to a data point, and each column represents a specific feature. The values are scaled between 0 and 1, where 0 corresponds to the minimum value of the feature and 1 corresponds to the maximum value.

8. from sklearn.model_selection import train_test_split

# Membagi dataset menjadi subset pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(normalized_data, y, test_size=0.2, random_state=42)

# Menampilkan hasil pemisahan dataset
print("Subset Pelatihan (X_train):", X_train)
print("Subset Pengujian (X_test):", X_test)
print("Label Pelatihan (y_train):", y_train)
print("Label Pengujian (y_test):", y_test)

Output :
Subset Pelatihan (X_train): [[0.24761905 0.56521739 0.16666667]
 [0.02698413 0.65217391 0.66666667]
 [0.4        0.17391304 1.        ]
 ...
 [0.14126984 0.82608696 1.        ]
 [0.2        0.52173913 0.16666667]
 [0.3968254  0.04347826 0.83333333]]
Subset Pengujian (X_test): [[0.0952381  0.43478261 1.        ]
 [0.77777778 0.2173913  0.83333333]
 [0.23809524 0.60869565 0.        ]
 [0.63174603 0.17391304 0.33333333]
 [0.66984127 0.17391304 0.16666667]
 [0.62222222 0.17391304 0.16666667]
 [0.08571429 0.69565217 1.        ]
 [0.97619048 0.30434783 0.66666667]
 [0.14285714 0.86956522 0.66666667]
 [0.14603175 1.         0.        ]
 [0.14920635 0.91304348 0.16666667]
 [0.2        0.65217391 0.16666667]
 [0.03333333 0.95652174 1.        ]
 [0.12222222 0.73913043 0.        ]
 [0.03492063 0.56521739 0.33333333]
 [0.26190476 0.73913043 0.83333333]
 [0.02222222 0.7826087  1.        ]
 [0.19066138 0.56521739 0.66666667]
 [0.03015873 0.47826087 0.33333333]
 [0.12063492 0.60869565 0.83333333]
 [0.17460317 0.91304348 0.33333333]
 [0.11111111 0.95652174 0.66666667]
 [0.74126984 0.2173913  0.16666667]
 [0.14285714 0.39130435 1.        ]
 [0.         0.34782609 0.33333333]
 [0.19047619 0.86956522 0.83333333]
 [0.13968254 0.56521739 1.        ]
 [0.21428571 0.43478261 0.16666667]
 [0.09365079 0.82608696 1.        ]
 [0.0952381  0.7826087  0.66666667]
 [0.05555556 0.86956522 0.33333333]
 [0.04761905 0.39130435 0.33333333]
 [0.22380952 0.56521739 0.16666667]
 [0.02063492 0.60869565 0.33333333]
 [0.11904762 0.34782609 1.        ]
 [0.11904762 0.86956522 0.66666667]
 [0.52380952 0.08695652 0.66666667]
 [0.04761905 0.82608696 0.66666667]
 [0.42857143 0.04347826 0.66666667]
 [0.08253968 1.         0.33333333]
 [0.03650794 0.69565217 0.33333333]
 [0.16666667 0.69565217 0.83333333]
 [0.21746032 1.         0.        ]
 [0.04761905 0.52173913 0.        ]
 [0.21428571 0.73913043 0.83333333]
 [0.58730159 0.13043478 0.83333333]
 [0.44761905 0.17391304 1.        ]
 [0.09047619 0.         1.        ]
 [0.02380952 0.73913043 0.66666667]
 [0.05873016 0.95652174 0.33333333]
 [0.27142857 0.         0.        ]
 [0.91428571 0.30434783 0.        ]
 [0.95555556 0.30434783 0.16666667]
 [1.         0.30434783 0.66666667]
 [0.11904762 0.43478261 1.        ]
 [0.15555556 0.7826087  0.33333333]
 [0.44444444 0.08695652 0.83333333]
 [0.69365079 0.17391304 0.16666667]
 [0.04761905 0.91304348 0.83333333]
 [0.16666667 0.86956522 0.66666667]
 [0.01269841 0.39130435 0.66666667]
 [0.07142857 0.82608696 0.66666667]
 [0.11269841 0.7826087  0.        ]
 [0.33333333 0.         0.66666667]
 [0.07142857 0.7826087  0.66666667]
 [0.00952381 0.56521739 0.16666667]
 [0.0015873  0.56521739 0.83333333]
 [0.11904762 0.52173913 0.83333333]
 [0.11904762 0.39130435 0.16666667]
 [0.1031746  0.86956522 0.33333333]
 [0.01269841 0.95652174 0.16666667]
 [0.11111111 0.47826087 0.        ]
 [0.54761905 0.08695652 0.66666667]
 [0.         0.82608696 0.66666667]
 [0.02380952 0.91304348 0.83333333]
 [0.7        0.2173913  0.        ]
 [0.22380952 0.         0.        ]
 [0.47142857 0.17391304 1.        ]
 [0.13333333 0.73913043 1.        ]
 [0.01587302 0.91304348 0.66666667]
 [0.01587302 0.39130435 0.83333333]
 [0.78888889 0.2173913  0.16666667]
 [0.11904762 0.95652174 0.83333333]
 [0.15079365 0.91304348 0.33333333]
 [0.17619048 0.52173913 0.16666667]
 [0.43809524 0.08695652 0.        ]
 [0.04444444 0.65217391 0.33333333]
 [0.5        0.08695652 0.66666667]
 [0.03333333 0.60869565 0.16666667]
 [0.05238095 0.86956522 1.        ]
 [0.3047619  0.13043478 1.        ]
 [0.19047619 0.91304348 0.66666667]
 [0.25079365 0.         0.33333333]
 [0.11587302 0.65217391 0.33333333]
 [0.16984127 1.         0.        ]
 [0.56349206 0.13043478 0.83333333]
 [0.17619048 1.         1.        ]
 [0.37301587 0.04347826 0.83333333]
 [0.13492063 0.43478261 0.83333333]
 [0.00793651 0.82608696 0.33333333]
 [0.16825397 0.60869565 0.83333333]
 [0.11904762 0.56521739 0.        ]
 [0.19047619 0.47826087 1.        ]
 [0.02380952 0.82608696 0.66666667]
 [0.0952381  0.39130435 0.16666667]
 [0.14761905 0.91304348 1.        ]
 [0.01904762 0.         1.        ]
 [0.13015873 0.60869565 0.33333333]
 [0.12698413 0.86956522 0.33333333]
 [0.13174603 1.         0.16666667]
 [0.36984127 0.04347826 0.33333333]
 [0.14285714 0.7826087  0.66666667]
 [0.0984127  0.69565217 0.        ]
 [0.06825397 0.56521739 1.        ]
 [0.0047619  0.86956522 1.        ]
 [0.14603175 0.73913043 0.66666667]]
Label Pelatihan (y_train): 533    0.0
159    0.0
304    0.0
156    0.0
111    0.0
      ... 
167    0.0
202    0.0
366    0.0
531    0.0
198    0.0
Name: wearing_off, Length: 460, dtype: float64
Label Pengujian (y_test): 330    0.0
214    0.0
442    0.0
594    0.0
498    0.0
      ... 
171    0.0
451    0.0
340    0.0
368    0.0
164    0.0
Name: wearing_off, Length: 116, dtype: float64

= The code you provided demonstrates the usage of the `train_test_split` function from the `sklearn.model_selection` module. This function is commonly used in machine learning tasks to split a dataset into training and testing subsets.

Here's a brief explanation of the code =

1. The `train_test_split` function is imported from the `sklearn.model_selection` module.
2. The dataset, `normalized_data`, is divided into features (`X`) and labels (`y`).
3. The `train_test_split` function is called with the following parameters:
   - `normalized_data`: The features of the dataset.
   - `y`: The labels of the dataset.
   - `test_size`: The proportion of the dataset to include in the testing subset. In this case, it is set to 0.2, which means 20% of the data will be used for testing.
   - `random_state`: The seed value used by the random number generator. It ensures that the same random splitting is reproducible.
4. The function returns four subsets: `X_train`, `X_test`, `y_train`, and `y_test`.
   - `X_train`: The training subset of the features.
   - `X_test`: The testing subset of the features.
   - `y_train`: The training subset of the labels.
   - `y_test`: The testing subset of the labels.
5. The print statements display the values of the subsets (`X_train`, `X_test`, `y_train`, and `y_test`).

In summary, the code splits the `normalized_data` into training and testing subsets, where 80% of the data is used for training (`X_train` and `y_train`), and 20% is used for testing (`X_test` and `y_test`).

9. from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np
# Membuat objek model regresi linier
model = LinearRegression()
# Melatih model dengan data pelatihan
model.fit(X_train, y_train)
# Memprediksi nilai target dengan data pengujian
y_pred = model.predict(X_test)
# Menghitung R-square
r2 = r2_score(y_test, y_pred)
# Menghitung mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Menampilkan hasil
print("Koefisien regresi:", model.coef_)
print("Intersep:", model.intercept_)
print("Prediksi nilai target:", y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-square:", r2)
# Visualisasi model
plt.scatter(X_train[:, 0], y_train, color='blue', label='Data Pelatihan')
plt.scatter(X_test[:, 0], y_test, color='red', label='Data Pengujian')
plt.plot(X_test[:, 0], y_pred, color='green', label='Model Regresi Linier')
plt.xlabel('Fitur')
plt.ylabel('Target')
plt.title('Regresi Linier Berganda')
plt.legend()
plt.show()

Output: 
![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/431f18a1-f81c-4c28-bcd8-93d3e31f821b)
= the code is to involve a regression model with three independent variables, an intercept term, and predictions for the dependent variable. The model's performance is assessed using the mean squared error and the R-squared value.
 
10. fig = plt.figure()
ax = fig.add_subplot(projection ='3d')

ax.scatter(X_train[:, 1], X_train[:, 2], y_train, label ='y', s = 5)
ax.legend()
ax.view_init(45, 0)

plt.show()

output : 
![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/e8e5c028-6a05-493f-b615-9e298f506628)

= Overall, this code generates a 3D scatter plot using the data from X_train and y_train, where X_train[:, 1] represents the x-coordinates, X_train[:, 2] represents the y-coordinates, and y_train represents the z-coordinates. The plot is displayed with a specific viewing angle of 45 degrees elevation and 0 degrees azimuth.

=========
Conclusion: Kesimpul prediksi hasil pemodelan menggunakan regrsi linear, menunjukan bahwa model berhasil menunjukan hasil yang akurat. Hal ini terbukti dari skor MSE sebesar 0.05 yang berarti model tersebut memiliki eror yang kecil
=========
