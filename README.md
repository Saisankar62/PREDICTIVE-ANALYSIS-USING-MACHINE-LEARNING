# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING
**COMPANY**: CODTECH IT SOLUTIONS
**NAME**: BALLEDA SAISANKAR
**INTERN ID**: CT08DF1488
**DOMAIN**: DATA ANALYTICS
**DURATION**: 8 WEEKS
**MENTOR**: NEELA SANTOSH

#Description of the task
I successfully created a predictive analysis using machine learning model. It is a process of building a predictive classification model using machine learning to predict customer churn based on the Telco Customer Churn dataset.Firstly import the libraries like pandas,os,Matplotlib & Seaborn,LabelEncoder / OneHotEncoder.The data set i download from the webbrowser (kaggle),and load using the pandas library.Data Preprocessing opeartions like Drop irrelevant columns,Convert target variable (Churn) to numeric i.e.. 'Yes' -> 1, 'No' -> 0,Converts all categorical columns to binaryformat using one-hot encoding , making them compatible with machine learning algorithms.
split features and target:
X (Features)--> Contains all the columns except the target variable (Churn).These are the input variables used to predict the outcome.
y (Target)--> Contains only the Churn column, which is the output or label we want to predict.
A train-test split (80-20) was applied to divide the dataset into training and testing sets, ensuring that the model could be trained effectively and evaluated on unseen data.
To enhance model performance and reduce dimensionality, Feature Selection was implemented using the SelectKBest method with chi-square scoring , selecting the top 20 most relevant features from the dataset that have the strongest relationship with the target variable (Churn).Retrieves the names of the features that were selected by the SelectKBest feature selection method (selected_features).
Initializes a Logistic Regression model and Trains the model based only the selected important features (X_train_selected) and their corresponding target values (y_train).
Prediction--> Uses the trained logistic regression model to predict whether customers in the test set will churn (1) or not churn (0).
Evaluation--> Compares the predicted values (y_pred) with the actual known values (y_test) to assess model performance.Here Accuracy of the model and Classification Report i.e..Precision,Recall ,F1-score.
visual representation of the model's performance using a confusion matrix , which shows how many predictions were correct or incorrect like true negative,false positive,false nagative,true positive.Labels for the confusion martix are X-axis : Model predictions (Predicted),Y-axis : Actual values from the dataset (Actual).finally the result will be show in a heatmap of the confusion matrix added axis labels and a title.
The machine learning techniques like Classification,LogisticRegression,Feature Selection,Train-Test Split,Evaluation Metrics and the  tools are used vscode,os,fileformate.


#output

Accuracy of the model

![Image](https://github.com/user-attachments/assets/13d5b35b-ec7b-4226-8b80-1cb0dfa0727e)



Confusion matrix

![Image](https://github.com/user-attachments/assets/33c31870-cea1-46b1-b770-67efc7c2f0f2)
