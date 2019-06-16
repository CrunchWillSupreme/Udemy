# Eclat
setwd("C:/Users/willh/Udemy/Machine Learning A-Z/P14-Machine-Learning-AZ-Template-Folder/Part 5 - Association Rule Learning/Section 29 - Eclat")

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) # use this to make sparse matrix
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
# for support value, 3*7/7500 - we want items that were purchased 3 times a day, 7 days a week divided by the total number of transactions - 7500
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2)) #minlen, we want at least 2 items per group

# Visualizing the results
inspect(sort(rules, by = 'support')[1:10])

