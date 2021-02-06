# Assignment3Part1.py

# Sai Madhuri Yerramsetti
# August 13, 2020
# Student Number: 0677671
# Assignment3 Part1

# import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import MeanShift, estimate_bandwidth, Birch, KMeans
from sklearn.cluster import MiniBatchKMeans

# Get data
data = pd.read_csv('D:/Madhuri/Data science with python/Assignment 3/abalone.csv')

######################## Data investigation and preprocessing ############################

#Investigate the data types and summay statistics of data
print(data.head(5))
print("Dimensions of data:\n", data.shape)
print("Information about data types present in data:\n", data.info())
with pd.option_context('display.max_columns', 40):
    print("Summary statistics of data:\n", data.describe())
print("Covariation matrix of the variables:\n", data.cov())

# Add age column
data['Age'] = data['Rings'] + 1.5

#Check for any null values in all the columns
print("Number of null values in each column:\n", data.isnull().sum())

#Check for the categorical column values counts
print("Value counts in 'Sex' column:\n", data['Sex'].value_counts(dropna=False))
print("Value counts in 'Rings' column:\n", data['Rings'].value_counts(dropna=False))

# Check the value counts for Age column
print("Value counts in 'Age' column:\n", data['Age'].value_counts(dropna=False))

#Change the categorical column values into numerical values
dataEncoded  = pd.get_dummies(data['Sex'], prefix = 'Sex')
dfEncoded = pd.concat([data, dataEncoded], axis=1)
del dfEncoded['Sex']

# Check the Encoded Sex column
print(dfEncoded.head(5))

# Make a copy of this data frame to use it for heat map visualization
dfCorr = dfEncoded.copy()

#Discretize the Age column into two bins
ageDiscretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')

# Stored the binned values in a new column
binData = ageDiscretizer.fit_transform(dfEncoded[['Age']])
dfEncoded['Age_binned'] = pd.Series(binData.T[0])

# Check the binned Age column and its value counts
print(dfEncoded.head(5))
print(dfEncoded['Age_binned'].value_counts(dropna=False))

################################ Visualization ###########################################

# draw a countplot of the 'Sex' column
sns.countplot(x='Sex', data=data)

# Plot a heat map
plt.figure(figsize=(15,10))
sns.heatmap(dfCorr.corr(), annot=True, linewidths=0.30, cmap='RdYlGn')

#draw a histogram of the new binned Age
dfEncoded.hist('Age_binned')

# set the style of the graph to white and plot a boxplot with labelled axes and title
sns.set(style="white")
plt.figure(figsize=(15,10))
sns.boxplot(x="Sex", y="Age",
        palette=["r", "b"],
        data=data)
plt.xlabel("Sex of the abalone")
plt.ylabel("Age of the abalone")
plt.title("Age vs Sex")

# set the style of the graph to ggplot and plot a jointplot with labelled axes and title
plt.style.use('ggplot')
graph = sns.jointplot(x="Length", y="Age", data=data)
graph.set_axis_labels("Length of the abalone", "Age of the abalone")
plt.title("Age vs Length")

# set the style of the graph to white and plot a lmplot with labelled axes and title
sns.set(style="white")
sns.lmplot(x="Shell weight", y="Age", hue="Sex",
           height=5, data=data)
plt.xlabel("Shell weight of the abalone")
plt.ylabel("Age of the abalone")
plt.title("Age vs Shell weight")

# plot a relplot with labelled axes and title
sns.set(style="white")
sns.relplot(x="Height", y="Age", palette="muted", data=data)
plt.xlabel("Height of the abalone")
plt.ylabel("Age of the abalone")
plt.title("Age vs Height")
# Show the plot
plt.show()

############################### Data preparation #########################################

# Select required columns
df = dfEncoded[['Height', 'Shell weight', 'Sex_I', 'Sex_F', 'Sex_M', 'Age_binned']]

# Create variables X which contains all the columns that are used for classification and y that contain target variable
y = df['Age_binned'].values
X = df.drop('Age_binned', axis=1).values

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

############################### Classification ############################################

######################################## SVM classification ###############################

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Instantiate the GridSearchCV object
svmGcv = GridSearchCV(pipeline, parameters, cv=10)

# Fit to the training set
svmGcv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = svmGcv.predict(X_test)

# Compute and print metrics
print("Accuracy with SVM: ", svmGcv.score(X_test, y_test))
print("Tuned Model Parameters of SVM: ", svmGcv.best_params_)
print("Classification report of SVM is: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of SVM is: \n", confusion_matrix(y_test, y_pred))

# Plot the confusion matrix after normalizing it
fig1 = plot_confusion_matrix(svmGcv, X_test, y_test, 
                        cmap=plt.cm.Blues)
fig1.ax_.set_title("Confusion Matrix plot")
print(fig1.confusion_matrix)
plt.show()

#################################### ADA Boost classification #############################

#Scale the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate classifier
adaClf = AdaBoostClassifier()

# Specify the hyperparameter space
parameters = {'n_estimators' : np.arange(1, 50)}

# Instantiate the GridSearchCV object
adaGcv = GridSearchCV(adaClf, parameters, cv=10)

# Fit to the training set
adaGcv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = adaGcv.predict(X_test)

# Compute and print metrics
print("Accuracy with AdaBoost: ", adaGcv.score(X_test, y_test))
print("Tuned Model Parameters of AdaBoost: ", adaGcv.best_params_)
print("Classification report of AdaBoost is: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of AdaBoost is: \n", confusion_matrix(y_test, y_pred))

# Plot the confusion matrix after normalizing it
fig2 = plot_confusion_matrix(adaGcv, X_test, y_test, 
                        cmap=plt.cm.Blues)
fig2.ax_.set_title("Confusion Matrix plot")
print(fig2.confusion_matrix)
plt.show()

############################### Clustering ################################################

# Data preparation for clustering by scaling it and removing categorical column 'Sex'
dataCopy = data.copy()
del dataCopy['Sex']
del dataCopy['Rings']
del dataCopy['Age']
print(dataCopy.describe().T)
reqData = dataCopy.to_numpy()
scaledData = preprocessing.scale(reqData)

#################################### Mini Batch KMeans Clustering #########################

# initialize the variable
SSE = []

# Fit the data with kmeans multiple times 
for i in range(1,9):
    minikmeans = MiniBatchKMeans(n_clusters=i, max_iter=300)
    # Fit kmeans to data
    minikmeans.fit(scaledData)
    # get sum of squared errors for each iteration
    SSE.append(minikmeans.inertia_)

# Plot the sum of squared errors for the each cluster count
plt.plot(range(1,9), SSE)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of squared errors")
plt.title("Sum of squared errors with respect to number of clusters for scaled data")
plt.show()

# Instantiate the clustering model
minikmeans = MiniBatchKMeans(n_clusters=3, max_iter=300)

# Fit and predict the data
minikmeans.fit(scaledData)
predictions = minikmeans.predict(scaledData)

# Scatterplot between two features to check the clustering
plt.scatter(scaledData[:,2], scaledData[:,6], c=predictions)
plt.xlabel("Height")
plt.ylabel("Shell weight")
plt.title("Clustering using MiniBatchKMeans clustering algorithm")
plt.show()

#################################### Birch Clustering ######################################

# Take multiple threshold values
thresholdRange = [0.01, 0.05, 0.15, 0.25, 0.3, 0.4, 0.5]

# Check the silhouette_score for each corresponding threshold value to 
# determine optimal threshold value by fitting the model repetedly
for i in thresholdRange:
    birch = Birch(threshold=i, n_clusters=2)
    birch.fit(scaledData)
    labels = birch.labels_
    silhouette_avg = silhouette_score(scaledData, labels)
    print("For threshold value", i, "average silhouette score is :", silhouette_avg)
  

# Check the silhouette_score for each corresponding no. of clusters to
# determine optimal number by fitting the model repetedly
for i in range(2,9):
    birch = Birch(threshold=0.15, n_clusters=i)
    birch.fit(scaledData)
    labels = birch.labels_
    silhouette_avg = silhouette_score(scaledData, labels)
    print("For number of clusters", i, "average silhouette score is :", silhouette_avg)

# Instantiate the clustering model
birch = Birch(threshold=0.05, n_clusters=2)

# Fit and predict the data
birch.fit(scaledData)
predictions = birch.predict(scaledData)

# Scatterplot between two features to check the clustering
plt.scatter(scaledData[:,2], scaledData[:,6], c=predictions)
plt.xlabel("Height")
plt.ylabel("Shell weight")
plt.title("Clustering using Birch clustering algorithm")
plt.show()

##################################### Mean Shift Clustering #################################

# Determine optimal bandwidth value
bandwidth = estimate_bandwidth(scaledData, quantile=0.2, n_samples=500)

# Instantiate the clustering model
mnShift = MeanShift(bandwidth=bandwidth)

# Fit and predict the data
mnShift.fit(scaledData)
predictions = mnShift.predict(scaledData)

# Scatterplot between two features to check the clustering
plt.scatter(scaledData[:,2], scaledData[:,6], c=predictions)
plt.xlabel("Height")
plt.ylabel("Shell weight")
plt.title("Clustering using Mean shift clustering algorithm")
plt.show()





