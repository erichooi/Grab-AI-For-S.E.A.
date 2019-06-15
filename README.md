# Grab - AI FOR S.E.A

![safety_challenge](fig/logo.webp)

## Safety 
![safety_challenge](fig/safety.webp)

Based on telematics data, how might we detect if the driver is driving dangerously?

This is a data science assignment where you are expected to create a **data model** from a given training dataset.

### Problem Statement
Given the telematics data for each trip and the label if the trip is tagged as dangerous driving, derive a model that can detect dangerous driving trips.

### Introduction
Grab has been proactively pushing to make transportation in SEA safer. As part of the effort, we want to identify dangerous drivings in a timely manner.

1. You can use the ["Ride Safely"](https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip) dataset provided by Grab.
2. You are expected to create a Data Model based on the ["Ride Safely"](https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip) dataset in order to solve the problem statement(s).
3. You should also provide step by step documentation on how to run your code. Our evaluators will be running your data models on a test dataset.

The given dataset contains telematics data during trips (bookingID). Each trip will be assigned with label 1 or 0 in a **separate label file** to indicate dangerous driving. Please take note that dangerous drivings are labelled per trip, while each trip could contain thousands of telematics data points. Participants are supposed to create the features based on the telematics data before training models.

| Field | Description |
| -- | -- |
| bookingID | trip id |
| Accuracy | accuracy inferred by GPS in meters |
| Bearing | GPS bearing in degree |
| acceleration_x | accelerometer reading in x axis (m/s2) |
| acceleration_y | accelerometer reading in y axis (m/s2) |
| acceleration_z | accelerometer reading in z axis (m/s2) |
| gyro_x | gyroscope reading in x axis (rad/s) |
| gyro_y | gyroscope reading in y axis (rad/s) |
| gyro_z | gyroscope reading in z axis (rad/s) |
| second | time of the record by number of seconds |
| Speed | speed measured by GPS in m/s |

### Dependency
The notebook required these external packages outside of default packages coming together after installing jupyter notebook. Do install these packages before running the notebook.
```sh 
$ pip install scikit-plot
$ pip install seaborn
$ pip install xgboost
$ pip install lightgbm
$ pip install dask
$ pip install dask-ml
```

### To generate the prediction score
First, clone the repository to local machine
```git
$ git clone https://github.com/erichooi/Grab-AI-For-S.E.A..git
```
Then you can choose to run this section (**Run for Test Data**) in the notebook (**safety-challenge.ipynb**) to get the test result
**OR**
 just running the python script (**predict.py**) to get the test result.
```sh
$ python predict.py -t <<test.csv>>
```

Noted that both will generate test result in file called **submission.csv**.