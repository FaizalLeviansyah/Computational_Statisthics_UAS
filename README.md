![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/6253f5e6-e740-4028-9203-d25c66f3662b)# Computational_Statisthics_UAS
Computational statistics is a branch of statistics that focuses on the development and use of computer algorithms to solve statistical problems. It is a rapidly growing field, as the increasing availability of computational power has made it possible to analyze large and complex datasets. we get the material from our lecturer named : Mr.Septian Eng

======================================================================
<Background>
  Over time, the medicine’s effective time shortens, causing discomfort among PD patients. 
Thus, PD patients and clinicians must monitor and record the patient symptom changes for adequate treatment. 
Parkinson's disease (PD) is a slowly progressive nervous system condition caused by the loss of dopamine-producing brain cells. 
It primarily affects the patient's motor abilities, but it also has an impact on non-motor functions over time. 
Patients' symptoms include tremors, muscle stiffness, and difficulty walking and balancing. 
Then it disrupts the patients' sleep, speech, and mental functions, affecting their quality of life (QoL). With the improvement of wearable devices, patients can be continuously monitored with the help of cell phones and fitness trackers. We use these technologies to monitor patients' health, record WO periods, and document.
the impact of treatment on their symptoms in order to predict the wearing-off phenomenon in Parkinson's disease patients.
========================================================================
========================================================================
<Task>
Using multiple linear regression, try to find the best regression model that is suitable to represent the case: 
helping the doctors to create specific treatment strategies to manage Parkinson's disease and its associated symptoms properly. 
It means you are asked to create a model that can anticipate the "wearing-off" of anti-Parkinson Disease medication.
========================================================================

(StatKom_UAS_.ipynb)
Description 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
import csv
from google.colab import auth
from google.colab import files
from  sklearn import linear_model
-above function is to import the library

(Meet 3 old files)
1. <in (Example_1.py)>
The code starts by importing the necessary libraries, which are numpy and pandas.
It then uses the np.array() function to construct a NumPy array called new_data. The array has the following numbers: [23, 56, 45, 65, 59, 55, 62, 54, 85, 25, 55]. The code then generates a DataFrame by calling the pandas library's pd.DataFrame() function. It passes the new_data array to the function as an argument. The DataFrame is a two-dimensional tabular data structure capable of storing and manipulating data.
Finally, the code invokes the DataFrame's describe() method, which creates descriptive statistics for the data. The describe() method computes statistics like count, mean, standard deviation, minimum, 25th percentile (Q1), median (50th percentile or Q2), 75th percentile (Q3), and max values dataframe.

2. <in (Mode.py)>
it is to import the scipy.stats and statistics modules required for statistical calculations. It then specifies a range of values [70, 80, 70, 70, 90, 100] for which various statistical measures will be calculated.
  - From scipy.stats, stats.mode(value),
    Using the mode() method from the scipy.stats module, this function computes the mode of the given numbers. The mode is the value or values that appear the most frequently in the dataset. A ModeResult object is returned as the result.
  - The statistics module's s.mode(value),
    This function computes the mode of the given data using the statistics module's mode() function, which is a native Python module. The mode() function returns the most frequently occurring value in the dataset. If many values have the same peak frequency. 

3. <in (Native.py)>
   - ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/43e59706-4f4d-47e7-aaa1-38a685ed3dee)
   it is to initializes a list called apel with ten elements representing the weights of different apples.
   - ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/8630d3e1-1c27-41a3-b835-1344ab7cdeb2)
   it is initializes a variable weight_total to keep track of the total weight of all the apples.
   - ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/b0be70c0-3d40-41d0-9477-d77321f2ae31)
   it is Using the range function, this loop iterates over each entry in the apel list. Each element is added to the weight_total variable, resulting in the total weight of all the apples.
   - ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/abec155d-3d0a-4486-81bd-2cf11f2dbd6a)
  it is After the loop, the average weight of the apples is calculated by dividing the weight_total by the length of the apel list. The result is stored in the variable avg_apel and then printed.

Next, let's move on to the median calculation.
  - ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/ab4036d9-f61b-4b90-8816-d46c4fd9aa87)
  it is line imports the math module, which provides mathematical functions needed for the median calculation.
  -![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/3e92b420-0a0c-45ef-8620-a133a0d7fd20)
  it is  The bubblesort function uses the bubble sort algorithm to sort the components in ascending order. It takes a list called elements as input and runs through it numerous times, comparing neighboring components and exchanging them if necessary to bring larger elements to the end of the list. Following that, the sorted list is returned.
  - ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/e33dea3f-ae7d-4172-ac56-dd9c397b41b5)
  The calc_median function computes the median of a set of numbers. It accepts a list of elements as input. To begin, it invokes the bubblesort function to sort the elements list in ascending order. The modulo operator% is then used to determine whether the length of the list is even or odd. If it is an even number, it computes the center position and returns the average of the two middle elements. If it is unusual, it returns the middle element.
  - ![image](https://github.com/FaizalLeviansyah/Computational_Statisthics_UAS/assets/91838778/5d03dc7f-3279-4bfd-8eea-cc346cb962f1)

