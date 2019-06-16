# Apriori
setwd("C:/Users/willh/Udemy/Machine Learning A-Z/P14-Machine-Learning-AZ-Template-Folder/Part 5 - Association Rule Learning/Section 28 - Apriori")

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# for support value, 3*7/7500 - we want items that were purchased 3 times a day, 7 days a week divided by the total number of transactions - 7500
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))
# for items purchased 4 times a day
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualizing the results
inspect(sort(rules, by = 'lift')[1:10])
 