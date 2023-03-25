1. Mean Absolute Error


![t31](https://user-images.githubusercontent.com/116189666/227714034-191dc4e6-102a-435f-b7d4-a6c4888e8b9f.jpg)

The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value of the absolute error loss or l1-norm loss.
If y^i is the predicted value of the i-th sample, and yi is the corresponding true value, then the mean absolute error (MAE) estimated over nsamples is defined as
 
 ![t32after31](https://user-images.githubusercontent.com/116189666/227714069-b9a9b1f9-e5cb-4a44-82ab-1c92e58b3118.jpg)

2. R² score, the coefficient of determination


![t33](https://user-images.githubusercontent.com/116189666/227714081-751a9f5d-bee9-48fb-b1e1-784e84fe0809.jpg)


![t34](https://user-images.githubusercontent.com/116189666/227714092-70832e9b-5567-4a2c-aa80-b5fda7ec1cf0.jpg)

The r2_score function computes the coefficient of determination, usually denoted as R².
It represents the proportion of variance (of y) that has been explained by the independent variables in the model. It provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.
 
 
 
Classification Metrics:



Classification is about predicting the class labels given input data. In binary classification, there are only two possible output classes. In multiclass classification, more than two possible classes can be present. I’ll focus only on binary classification.
A very common example of binary classification is spam detection, where the input data could include the email text and metadata (sender, sending time), and the output label is either “spam” or “not spam.” (See Figure) Sometimes, people use some other names also for the two classes: “positive” and “negative,” or “class 1” and “class 0.”
The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance. Some metrics might require probability estimates of the positive class, confidence values, or binary decisions values. Most implementations allow each sample to provide a weighted contribution to the overall score, through the sample_weight parameter.
1. Accuracy Score


 ![taccuracy](https://user-images.githubusercontent.com/116189666/227714116-8c156b5b-9f44-45ab-a9cd-1442882e8d2c.jpg)
 
 
![taccuracy2](https://user-images.githubusercontent.com/116189666/227714121-d0260ef0-2405-4a23-b927-82a8ea622ab2.jpg)

The accuracy_score function computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions.
In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
If y^i is the predicted value of the i-th sample and yi is the corresponding true value, then the fraction of correct predictions over nsamples is defined as
 
 
2. Confusion matrix



![t3confusion](https://user-images.githubusercontent.com/116189666/227714129-a0da9be9-93ca-4d5f-ae92-0be8ce73b310.jpg)

The confusion_matrix function evaluates classification accuracy by computing the confusion matrix with each row corresponding to the true class (Wikipedia and other references may use different convention for axes).
By definition, entry in a confusion matrix is the number of observations actually in group , but predicted to be in group . Here is an example:
 
3. Classification Report



![t3classi](https://user-images.githubusercontent.com/116189666/227714138-a0f7dd78-20eb-46b1-9a4e-5ab79d96d323.jpg)

The classification_report function builds a text report showing the main classification metrics. Here is a small example with custom target_names and inferred labels:
