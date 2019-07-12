# Artificial Neural Network

# Classification Template
setwd("C:/Users/willh/Udemy/Machine Learning A-Z/P14-Machine-Learning-AZ-Template-Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)")

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding categorical data
dataset$Geography = as.numeric(factor(dataset$Geography,
                                       levels = c('France', 'Spain', 'Germany'),
                                       labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11]) # 11 is the column with the dependent variable
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_211')
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              # number of hidden nodes in hidden layer is average of input nodes and output nodes.  
                              # Input nodes is 10 because we have 10 independent variables, output layer is 1 because we have a binary outcome.
                              # Number of nodes in hidden layer is 10+1/2 = 5.5, round to 6.  We'll have 2 hidden layers.
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)


# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
# Accuracy = number correct / total
(1535 + 195)/2000 # = 0.865

# disconnect from h2o server
h2o.shutdown()

