# K-Means Clustering
setwd("C:/Users/willh/Udemy/Machine Learning A-Z/P14-Machine-Learning-AZ-Template-Folder/Part 4 - Clustering/Section 24 - K-Means Clustering")

# Importing the mall dataset
dataset <- read.csv('Mall_customers.csv')
X <- dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type = "b", main = paste('Clusters of clients'), xlab = 'Number of clusters', ylab = 'WCSS')

# Applying k-means to the mall dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualizing the clusters - only for 2-d clustering
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         ylab = 'Spending Score')