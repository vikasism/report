My name is Vikas, student at Marvel, and today with all buzz I have started learning at MARVEL UVCE officially.

I am given a domain of AIML, and yes that’s what I had opted for! Here at marvel, we have levels to climb consisting of many tasks. Today I thought off of starting with level 1,but no sooner I visited the website, I realized there are some introduction concepts to study before getting on levels. Hence I started the first intro concept i.e. Statistics in AIML.

Statistics is all about collecting data, analyzing it and presenting it in an interpretable way. Its of two types: Descriptive and Inferential. Descriptive is all about summarizing data and presenting it using bar graphs, pie charts etc. Mean ,Median, Mode, Standard Variance etc are used in this type. Inferential statistics is about considering sample data and drawing conclusions about it. Concept of Probability is used here.

Why should u learn Statistics for ML?

-   Say, you have developed a product, Statistics tells you what features are important in your product.
-   It tells about the minimal performance required and how you can design strategy for your product.
-   It also tells the most common and expected outcome of your product.
-   So Statistics is a tool for you to be on a safer side.

# Task1

The data is imported as a pandas dataframe, then we add the house prices to it. This will allow us to manipulate this data mathematically by using inbuilt methods in pandas which is called data cleaning.

We will start with importing basic modules such as numpy, pandas and matplotlib. We put the data into a data frame which behaves like an 2D array in C, using Pandas. With data frames we can manipulate the data mathematically by using inbuilt methods in pandas which is called data cleaning.

We will use a set of data that helps us understand the relationship between variables. We import numpy to convert the data from Pandas into a 2D array. Then we can manipulate the data mathematically by adding columns and slicing it in different ways with pd.DataFrame().

Linear Regression is a supervised learning algorithm, which means that it works on a set of labeled examples. Before we look at how to use linear regression in Python, let's first learn about the concepts we need to understand before using the appropriate methods,

The steps involved in implementing the model are as follows:

1.  Initially the libraries are imported

    ![](media/6a906c4b39d5c67ae1d5001508a68ed7.png)

2.  The Boston housing price data is loaded from the datasets. The target contains the prices of the houses feature_names contain the names of all the features in the dataset (except target variables). Before moving on to the next step, the data is checked for any missing values i.e the null values. There are no missing values in the dataset

    ![](media/4b2c810e7887acb3741471477d70ab1e.png)

3.  The data splitted as training data and testing data The size value 0.2 indicates that 80% of the samples are used to train the model and the remaining 20% to evaluate or test the model. For this purpose, we use we utilize the scikit learn library's train_test_split() function as shown

    ![](media/0b6208f38fb632a4f21b31ea9256e80f.jpg)

4.  Scikit learn's LinearRegression() is used to train our model on both the training and test sets

    ![](media/a6158f045625166caa785e2bbc871dcf.jpg)

5.  This step is to determine whether the model is overfitting or not. Overfitted models perform well on test datasets but do not predict well on real-world datasets The root mean squared error(rms) and R2 score is used to assess the overfitting. The R2 score is of both training and testing data are almost equal to one. Hence the model is not overfitting

    ![](media/0671b72ffd16d6ed4df77fdaa19cfa52.jpg)

6.Finally the actual prices vs predicted prices are visualized using scatter plot

![](media/955cfb6538da6b8fb63a0d53f95b84ab.jpg)

## Task1(2)

In this project we are going to learn Logistic Regression. Logistic regression is modelling the probability of a discrete outcome given an input variable. It says whether something is true or false. The prediction logistic regression is categorical because it will have different classes in iris dataset present in iris.target, and different features based on which they are divided are present in iris.feature_names

Logistic regression is used to predict the probability of a discrete outcome given an input variable. In this project, our goal is to optimize the performance of logistic regression by minimizing bias and variance. This can be done by finding out the model with best results on training set subset, while also performing cross validation on testing set.

Logistic Regression is a supervised classification algorithm. Logistic regression measures the relationship between one or more independent variables (X) and the

categorical dependent variable (Y) by estimating probabilities using a logistic(sigmoid) function. Linear regression is not used for classification The steps involved in implementing the model are as follows:

1.  Initially the libraries are imported

    ![](media/645bf4b81b38c7221d12adf33bb014b4.jpg)

2.  The iris dataset is loaded from the Iris.csv file stored in the location

    ![](media/93f01c85b84cea23adc7c034aa89ac4a.jpg)

3.  The four features, SepalLength, SepalWidth, PetalLength, and PetalWidth. The last feature, 'Species,' is the target feature we will predict

    ![](media/28266fe29e1c550bb2ce08afba2e5be7.jpg)

4.  Before moving on to the next step, the data is checked for any missing values

    i.e the null values

    ![](media/15b7e099594d4fe1b00cb39a9cc8d2f2.jpg)

There are no missing values in the dataset

1.  The facegrid from seaborn library is used visualize the Petal length vs Petal width for and Sepal length vs Sepal width for different species

    ![](media/5b7bbacfc3994eaacf110fc55ebaeb15.jpg)

2.  All the data excluding the ID and species are taken as training and testing data

    ![](media/37851ef329d58963ebc3ed1f72a5a767.jpg)

3.  The data splitted as training data and testing data. The size value 0.2 indicates that 80% of the samples are used to train the model and the remaining 20% to evaluate or test the model. For this purpose, we use we utilize the scikit learn library's train_test_split() function as shown

    ![](media/8614ab03622f1cd759ad380129ab205b.jpg)

4.  Scikit learn's LogisticRegression() is used to train our model on both the training and test sets. The accuracy_score function imported from metrics library is used to calculate the accuracy of the model

    ![](media/5ef175b754144c9e1ff2c94d48650e0f.jpg)

## TASK3

Some of the plot which are plotted using matplotlib.pyplot and seaborn are shown here,

1.Ball and Bubble plot

2.Bar plot

3.pie plot

4.Violin plot

5.3D plot

EXAMPLE IMAGES FOLLOWS:

![](media/d7552aae47b557ddd353c0952ed0b2ab.jpg)

![](media/5ef1835a91d629a574a052d673a9b005.jpg)

### Matplotlib and Visualizing Data Matplotlib

![](media/ca349c6e4b5e4655810118060224e5aa.png)It is used for basic graph plotting like line charts, bar graphs, etc. It mainly works with datasets and arrays. Matplotlib acts productively with data arrays and frames. It regards the aces and figures as objects. It is more customizable and pairs well with Pandas and Numpy for Exploratory Data Analysis.

The different types of plots which can be used to visualize the dataset are as follows:

**Line and area plot**

#### Scatter and Bubble Plot

![](media/227ce917c4cbd50a244793056d8db6b7.jpg)

![](media/71df6a60f46610965b0dba16fc82adec.png)**Simple Bar Plot**

![](media/d4f63931f638b7b07ef4788e22177360.jpg)

**Grouped bar plot**

### Stacked bar plot and histogram

![](media/9eafad3640ef6baebf60bc067b8095cd.jpg)

**Pie plot**

![](media/a3619e672d809e573b38085898e292d7.jpg)

**Box plot**

![](media/9eef6fb22a8579d9af95ad06be4d13f3.jpg)

**Marginal plot**

![](media/8d60a77ab35afc6c52c80ec65149a46a.jpg)

### Heat map and 3D plot

![](media/870499d4806c6e35dc0b0a187ed0f2ca.jpg)

## TASK3

Linear regression is a commonly used statistical method that allows us to model the relationship between a dependent variable and one or multiple independent variables. In this report, we will discuss how to implement linear regression from scratch using example images.

Step 1: Preparation of data

We will use example images in the form of arrays, where each image is represented as a two-dimensional array with rows and columns representing pixels. The dependent variable will be the label associated with each image, and the independent variables will be the pixel values. We will divide the data into a training set and a testing set.

Step 2: Initialization

We need to initialize the weights and biases of our linear regression model. These values will be updated during the training process to minimize the loss function.

Step 3: Training

The training process involves iterating over the training data and updating the weights and biases to minimize the loss function. We will use mean squared error as the loss function, which measures the difference between the predicted output and the actual label. The gradient descent algorithm can be used to update the weights and biases in the direction of the minimum loss.

Step 4: Testing

Once the training process is completed, we can evaluate the performance of the model on the testing data. We will calculate the mean squared error on the test data and use it as a measure of how well the model has learned the relationship between the pixel values and the labels.

Step 5: Deployment

Finally, we can deploy the model in a real-world scenario by using it to predict the label of new images based on their pixel values.

Regression Metrics:

In regression problems, we need to evaluate the performance of the model by comparing the predicted values to the actual values. The following regression metrics will be used in this project:

Mean Absolute Error (MAE) - measures the average magnitude of the errors in a set of predictions.

Mean Squared Error (MSE) - measures the average of the squares of the errors in a set of predictions.

Root Mean Squared Error (RMSE) - the square root of the MSE.

Evaluation:

The performance of each algorithm will be evaluated based on the accuracy of the predictions on the test data. The accuracy is defined as the proportion of correct predictions in the test data.

### 1. Mean Absolute Error

The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value of the absolute error loss or l1-norm loss.

If y\^i is the predicted value of the i-th sample, and yi is the corresponding true value, then the mean absolute error (MAE) estimated over nsamples is defined as

![](media/626aef155a1d0567da5b83d8ba8599b2.jpg)

![](media/636aa624e814310ed0037de84982e116.jpg)

### 2. R² score, the coefficient of determination

The r2_score function computes the coefficient of determination, usually denoted as R².

It represents the proportion of variance (of y) that has been explained by the independent variables in the model. It provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.

![](media/a1d666806382b6378800180b47c3a316.jpg)

![](media/c7198113e371f0224292e49967b4763c.jpg)

**Classification Metrics:**

Classification is about predicting the class labels given input data. In binary classification, there are only two possible output classes. In multiclass classification, more than two possible classes can be present. I’ll focus only on binary classification.

A very common example of binary classification is spam detection, where the input data could include the email text and metadata (sender, sending time), and the output label is either “spam” or “not spam.” (See Figure) Sometimes, people use some other names also for the two classes: “positive” and “negative,” or “class 1” and “class 0.”

The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance. Some metrics might require probability estimates of the positive class, confidence values, or binary decisions values. Most implementations allow each sample to provide a weighted contribution to the overall score, through the sample_weight parameter.

### 1. Accuracy Score

The accuracy_score function computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions.

In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

If y\^i is the predicted value of the i-th sample and yi is the corresponding true value, then the fraction of correct predictions over nsamples is defined as

![](media/f0c65e6a0bfc18eab5b26102e10e2b1c.jpg)

![](media/631c615a51cc4b28a7d2f35eb57d91b5.jpg)

### 2. Confusion matrix

The confusion_matrix function evaluates classification accuracy by computing the confusion matrix with each row corresponding to the true class (Wikipedia and other references may use different convention for axes).

By definition, entry in a confusion matrix is the number of observations actually in group , but predicted to be in group . Here is an example:

![](media/15b1fabe2e640fdad13ec53d61442f56.jpg)

### 3. Classification Report

The classification_report function builds a text report showing the main classification metrics. Here is a small example with custom target_names and inferred labels:

![](media/a2eb7ac9bf8a6b6196cfd566b4f267f7.jpg)

### TASK4

**Linear and Logistic Regression - Coding the model from SCRATCH**

#### Linear Regression

Implementation of linear regression model on a particular dataset involves various functions. The key concepts involved in deriving the linear regression model are as follows

Given our simple linear equation y=mx+b, we can calculate MSE as:

![](media/4ae78452a5214513bb763056621dab1d.jpg)

We can calculate the gradient of this cost function as:

![](media/f3eaa93b4be0ad5a05df0a1c608a1c29.jpg)

The steps involving in implementation of the linear regression model are as follows:

1.  Firstly we create a class LinearRegression. In the init function we set the learning the rate value to 0.001 and number of iterations to default value 1000

    ![](media/57fb3eb5ee066c5cd91152df88066072.jpg)

2.  Now we define a fit method which takes the training samples and labels associated with them and the gradient descent function is implemented in this step as per the formula specified

    ![](media/8d985cc83670e5557ae45dc6d98e1a09.jpg)

3.  Then the predict method is defined which gets the new test samples

    ![](media/b504e0db5d4b6f52b0d6c512e2b8b169.jpg)

4.  The necessary libraries are imported

    ![](media/8233cfa43c38919f745c3732b10e5a30.jpg)

5.  The data splitted as training data and testing data The size value 0.2 indicates that 80% of the samples are used to train the model and the remaining 20% to evaluate or test the model. For this purpose, we use we utilize the scikit learn library's train_test_split() function as shown

    ![](media/3b9727cf44251a5159ea4ec8af123fa1.jpg)

6.  The training data and testing data are fit into the dataframe and the test values are predicted

    ![](media/bb78e2b6d7c45f77f3c07e3bbf160818.jpg)

7.  The cost function is defined as mse to calculate the mean squared error and the accuracy of the model is calculated

    ![](media/121ae2198e7003cbac780f95f95e9ff4.jpg)

8.  The scatter plot is used to visualize the data which has been approximated

#### LOGISTIC REGRESSION

The steps involving in implementation of the linear regression model are as follows:

1.  Firstly we create a class LinearRegression. In the init function we set the learning the rate value to 0.001 and number of iterations to default value 1000

    ![](media/ac4cb78c3bf39c1ae7379ad7969adb50.jpg)

2.  Now we define a fit method which takes the training samples and labels associated with them and the gradient descent function is implemented in this step as per the formula specified

    ![](media/2e234eeb58caf813cf3a0eae82bd4d03.jpg)

3.  Then the sigmoid function is defined as per the formula specified

    ![](media/994f5271d3a7fac41421351d1a382df1.jpg)

4.  Then the predict method is defined which gets the new test samples and approximate the values as per a linear model and uses sigmoid function. It predicts 1 if the value is greater than 0.5 and 0 if the value is less than 0.5

    ![](media/18b085db3ea560eaa4f484b60919d75f.jpg)

5.  The necessary libraries are imported and the iris dataset is loaded

    ![](media/5441340de964a2f4b2833985d052ab7f.jpg)

6.  The data splitted as training data and testing data

    The size value 0.2 indicates that 80% of the samples are used to train the model and the remaining 20% to evaluate or test the model. For this purpose, we use we utilize the scikit learn library's train_test_split() function as shown

    ![](media/ea25128637a8d8948d581d4bb0bef254.jpg)

7.  The training data and testing data are fit into the dataframe and the test values are predicted

    ![](media/b8bd73dd91a4165d77c6e3fd6bbb0ba1.jpg)

8.  Lastly the accuracy function is used to obtain the accuracy of the model

    ![](media/f18a349839551a32a389beb49580b79d.jpg)

    In conclusion, implementing linear regression from scratch with example images requires a few simple steps: preparation of data, initialization, training, testing, and deployment. The trained model can be used to make predictions on new data and is a powerful tool for solving real-world problems

### TASK 6

The k-Nearest Neighbor (k-NN) algorithm is a popular and simple machine learning algorithm used for classification and regression problems. It is a non-parametric algorithm that makes predictions based on the majority of the k-nearest data points to a given sample.

Working of k-NN:

The k-NN algorithm operates on the principle of instance-based learning. It uses a distance metric, such as Euclidean distance, to calculate the similarity between a given sample and the data points in the training set. The k-nearest data points to the sample are selected, and their class labels are used to make a prediction.

In classification problems, the majority vote of the k-nearest data points is used to predict the class label of a sample. In regression problems, the average of the k-nearest data points is used to make a prediction.

Selection of k:

The choice of k is a crucial factor in the k-NN algorithm as it affects the performance of the algorithm. A small value of k results in a high variance model, which is sensitive to noise in the data. A large value of k results in a high bias model, which may miss important patterns in the data.

Example:

Let's consider a binary classification problem, where we need to classify a given data point as either class 0 or class 1. The k-NN algorithm is applied to a sample data set with two features, x1 and x2, as follows:

The sample data set is split into a training set and a test set.

The k-NN algorithm is trained on the training set and k is set to 3.

The k-NN algorithm is used to make predictions on the test set by finding the 3 nearest data points in the training set and taking the majority vote of their class labels.

The accuracy of the predictions is evaluated and compared to other algorithms.

**Implement KNN using sci-kit’s neighbors.KNeighborsClassifier for multiple suitable dataset**

The steps involved in implementing scikit learn library are as follows:

1.  Initially the libraries required are imported

    ![](media/72b5df84b1b4755685cb54f5cb43d14d.jpg)

2.  The iris dataset is loaded from the Iris.csv file stored in the location

    ![](media/952304da1bb888818cf59acf97466c02.jpg)

3.  All the features and their values can be displayed as shown

    ![](media/d4f508d6769c0dee119e3eb36325903d.jpg)

4.  Since the species has many null values it cannot be considered for training and testing. Hence except that column and its values the rest are considered and stored in x

    ![](media/3fb37f7f9cc9f843d313621c377cdb3a.jpg)

5.  The species column with names are considered and stored in y

    ![](media/32d628b4d30389acc60097381854a154.jpg)

6.  To locate the data point in multidimensional feature space, it would be helpful if all features are on the same scale. Hence normalization or standardization of data will help

    ![](media/952a8284181ca1c69a3d740d455b8ef0.jpg)

7.  The data splitted as training data and testing data The size value 0.3 indicates that 70% of the samples are used to train the model and the remaining 30% to evaluate or test the model. For this purpose, we use we utilize the scikit learn library's train_test_split() function as shown

    ![](media/361ad4c70532123b1ca3fd3bedb58299.jpg)

8.  The accuracy function is used to calculate the accuracy of the model

    ![](media/6786b3527c09af2a205270f24c5e99a6.jpg)

9.  The confusion matrix is imported from the library and gives us an idea about which of the values are not classified properly

    ![](media/19067e8cb941cca6c9a3d62e2eb30b22.jpg)

**Implement KNN from scratch. Compare results with sci-kit’s built in method for different datasets.**

Implementation of KNN algorithm from scratch involves the following steps:

1.  The euclidean_distance function is defined which computes the distances from a specific as described in the formula

    ![](media/e7f127918cd1fd25b63e067771289aba.jpg)

2.  The KNN class is defined. It consists of fit, predict and \_predict method. The fit method stores the training samples. The predict method stores multiple samples of data whereas the \_predict method takes only one data sample. \_predict method computes the distances and gets the k nearest samples and also the most common class label.

    ![](media/a6a4fa073fa3b44209885ac5b8af001d.jpg)

3.  The necessary libraries are imported which are required for the implementation of the model and the iris dataset is loaded from the datasets. X and y have the data values and target values respectively

    ![](media/9fca2f173bd7e05bde72676a89d7818a.jpg)

    ![](media/3279364f0197433e2ee44c5ec9660302.jpg)

4.  The data splitted as training data and testing data The size value 0.2 indicates that 80% of the samples are used to train the model and the remaining 20% to evaluate or test the model. For this purpose, we use we utilize the scikit learn library's train_test_split() function as shown

    ![](media/0a0cc86f8f8184a9ede47b94759216a6.jpg)

5.  The data is then fit into the data frame and the accuracy is calculated

    ![](media/f25262ef5fcd977794746a075c23e5e6.jpg)

    When compared both the models are returning approximately the same value

    Conclusion:

    The k-NN algorithm is a simple and effective algorithm that can be used for both classification and regression problems. Its performance depends on the choice of k and the distance metric used. In practice, the k-NN algorithm is often used in combination with other algorithms for improved performance
