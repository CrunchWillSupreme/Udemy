# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:25:00 2019

@author: willh
"""

# Apriori
import os
os.chdir(r'C:\Users\willh\Udemy\Machine Learning A-Z\P14-Machine-Learning-AZ-Template-Folder\Part 5 - Association Rule Learning\Section 28 - Apriori')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
# build list of lists
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# Training Apriori on the dataset
from apyori import apriori
# for support, consider items that are purchased 3 times a day, 7 days a week= 21, divided by total number of transactsion, 7500. 21/7500 = 0.0028 ~ 0.003
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing the reesults
results = list(rules)
