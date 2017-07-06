# Building Classification Models for Human Resources Analytics

## Dataset source:
https://www.kaggle.com/ludobenistant/hr-analytics

## What I tried to do

The objective of this project was to make a classification model which could accurately predict whether or not an employee will leave a company prematurely. Models were evaluated on a simulated dataset which covered an employee's:

* Last evaluation score
* Number of projects
* Average monthly hours
* Time spent at the company
* Whether they have had a work accident
* Whether they have had a promotion in the last 5 years
* Department
* Salary
* Whether the employee has left

A 70-30 training-test data split was used for model training. Performance was evaluated based on a confusion matrix. The best results were yielded by a decision tree classifier which had 97.40%. The parameters which gave the best results were Gini impurity and a maximum of 40 bins used for discretizing continuous features and determining where to split on features.

A snapshot of the best confusion matrix is provided and you can see the results of different result combinations for the classifier models that were experimented with in txt files
