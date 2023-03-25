**TASK1(2)**


The steps involved in implementing the model are as follows:
1.	Initially the libraries are imported
 ![task21](https://user-images.githubusercontent.com/116189666/227712568-d018e6cf-f4d8-49e9-991f-1f1cb16615cf.jpg)

2.	The iris dataset is loaded from the Iris.csv file stored in the location
 ![task22](https://user-images.githubusercontent.com/116189666/227712575-5c386c43-742d-4288-b605-8e50f56372e2.jpg)

3.	The four features, Se![task23](https://user-images.githubusercontent.com/116189666/227712579-0eb7a629-2244-4938-9ebc-e69482638d1a.jpg)
palLength, SepalWidth, PetalLength, and PetalWidth. The last feature, 'Species,' is the target feature we will predict
 
4.	Before moving on to the next step, the data is checked for any missing values
i.e the null values![task24](https://user-images.githubusercontent.com/116189666/227712584-57e1e126-de6f-4d46-b3ba-161089624dfb.jpg)

 
There are no missing values in the dataset
5.	The facegrid from seaborn library is used visualize the Petal length vs Petal width for and Sepal length vs Sepal width for different species
 ![task25](https://user-images.githubusercontent.com/116189666/227712587-91ef5f89-3fba-4fc5-b9e8-e52705e6feec.jpg)

6.	All the data excluding the ID and species are taken as training and testing data
 ![task26](https://user-images.githubusercontent.com/116189666/227712591-74c88234-2c46-4ef6-b621-c406ad41e468.jpg)

7.	The data splitted as training data and testing data. The size value 0.2 indicates that 80% of the samples are used to train the model and the remaining 20% to evaluate or test the model. For this purpose, we use we utilize the scikit learn library's train_test_split() function as shown
 ![task27](https://user-images.githubusercontent.com/116189666/227712595-819441cf-90af-4be0-ad3a-7c55316d9dbe.jpg)

8.	Scikit learn's LogisticRegression() is used to train our model on both the training and test sets. The accuracy_score function imported from metrics library is used to calculate the accuracy of the model
![task28](https://user-images.githubusercontent.com/116189666/227712597-b4ef400a-5b57-423c-96ed-35d89ff9b8ff.jpg)
