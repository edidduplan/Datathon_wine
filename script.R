# ----------------------------------

# EXPLANATION 

# This dataset deals with our favourite beverage: wine.
# It is about a white variant of the Portuguese "Vinho Verde".
# Vinho verde is a unique product from the Minho (northwest) region of Portugal. 
# Medium in alcohol, it is particularly appreciated due to its freshness (specially in the summer).

# The goal is to model wine quality based on physicochemical tests and sensory data.
# There is no data about grape types, brand, selling price, etc.
# As the class is ordered, you could choose to do a regression instead of a classification.

# The prediction will have to be uploaded in one of your public repository in your github

# Deadline is at 5.30pm!

# For every minute of late delivery your final accuracy will be decreased by 0.05
# For example If your final accuracy is 86% but you deliver 20 minutes late then
# your final accuracy will be 0.86 - (0.05 x 20 = 1) = 0.85
#
# No extra point for early delivery.

# Let's start !

# ----------------------------------

# Import Libraries
library(rstudioapi)
library(ggplot2)
library(caret)
library(tidyverse)
pacman::p_load(caretEnsemble, corrplot, data.table)

# Load the data
setwd(dirname(getActiveDocumentContext()$path))
df <- read.csv("data/training.csv", sep = ",", header = TRUE)

# See row and columns numbers
dim(df)

# Look at the classes of the features
str(df)

# Look at the name of the features
names(df)
# Summary of the data
summary(df)


# Look for missing values
sum(is.na(df))
any(is.na(df))

# Cleaning the df
df$quality <- as.factor(df$quality)
df$X <- NULL

# Look at the distribution of the dependent variable
ggplot(df, aes(quality)) +
  geom_bar()  +
  labs(title=" Dependent variable distribution",
        x ="Quality of the wine",
        y = "")
  
ggplot(df, aes(quality)) +
  geom_histogram(stat = "count")

# Distribution of the predictors
ggplot(df, aes(fixed.acidity)) +
  geom_histogram()

ggplot(df, aes(volatile.acidity)) +
  geom_histogram()

ggplot(df, aes(citric.acid)) +
  geom_histogram()

ggplot(df, aes(residual.sugar)) +
  geom_histogram()

ggplot(df, aes(chlorides)) +
  geom_histogram()

ggplot(df, aes(free.sulfur.dioxide)) +
  geom_histogram()

ggplot(df, aes(total.sulfur.dioxide)) +
  geom_histogram()

ggplot(df, aes(density)) +
  geom_histogram()

ggplot(df, aes(pH)) +
  geom_histogram()

ggplot(df, aes(sulphates)) +
  geom_histogram()

ggplot(df, aes(alcohol)) +
  geom_histogram()

# Scatter plots
ggplot(df, aes(x=residual.sugar, y=density)) +
  geom_point()

ggplot(df, aes(x=alcohol, y=density)) +
  geom_point()

ggplot(df, aes(x=free.sulfur.dioxide, y=total.sulfur.dioxide)) +
  geom_point()

ggplot(df, aes(x=density, y=quality)) +
  geom_point()

ggplot(df, aes(x=alcohol, y=quality)) +
  geom_point()

ggplot(df, aes(x=chlorides, y=quality)) +
  geom_point()

# Exploratory Analysis
df_9 = filter(df, quality == 9)

summary(df_9)

#============= Correlation matrix ===================
df$quality <- as.integer(df$quality)
corr_matrix <- cor(df, method = "pearson")

corrplot(corr_matrix, tl.cex = 0.6, method = "pie")

corrplot(cor_existingprod, method = "pie")

#Colinearities between:
# - (density) - residual.sugar
# - (density) - alcohol
# - total.sulfur.dioxide - (free.sulfur.dioxide)

# Taking out outliers
# density outlier
df_clean <- filter(df, density < 1.01)

ggplot(df_clean, aes(x=chlorides, y=quality)) +
  geom_point()

ggplot(df_clean, aes(density)) +
  geom_histogram()

df_clean2 <- filter(df_clean, chlorides < .225)

ggplot(df_clean3, aes(x=free.sulfur.dioxide, y=quality)) +
  geom_point()

df_clean3 <- filter(df_clean2, free.sulfur.dioxide < 200)

df_9 = filter(df, quality == 9)

ggplot(df, aes(quality)) +
  geom_histogram(stat = "count")

# Taking out colinear predictors
df_clean4 <- df_clean3
df_clean4$density <- NULL
df_clean4$free.sulfur.dioxide <- NULL

#Checking duplicates
df_clean5 <- unique(df)
 ggplot(df_clean5, aes(quality)) +
   geom_histogram(stat = "count")

table(df_clean5$quality)

# Run model
t.control <- trainControl(method = "cv", number = 10)
model <- train(quality ~ ., data = df, methode = "rf", trcontrol = t.control)

model$results # ~ 65% accuracy

# Once you have your predicsion, load the validation and fill the quality feature
validation <- read.csv("data/validation.csv", sep = ",", header = TRUE)

print(validation$quality)

# Good luck