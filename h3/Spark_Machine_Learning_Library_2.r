# Databricks notebook source
# Installing sparklyr takes a few minutes,
# because it installs +10 dependencies.

if (!require("sparklyr")) {
  install.packages("sparklyr")
}

# Load sparklyr package.
library(sparklyr)

# COMMAND ----------

# create a sparklyr connection
sc <- spark_connect(method = "databricks")

# COMMAND ----------

library(dplyr)
head(iris)

# COMMAND ----------

str(iris)

# COMMAND ----------

iris_tbl <- sdf_copy_to(sc = sc, x = iris, overwrite = T)

# COMMAND ----------

str(iris_tbl)

# COMMAND ----------

iris_tbl

# COMMAND ----------

kmeans_model <- iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  ml_kmeans(features=c("Petal_Width", "Petal_Length"), k = 3)

# COMMAND ----------

# print our model fit
kmeans_model

# COMMAND ----------

kmeans_model$features

# COMMAND ----------

kmeans_model$centers

# COMMAND ----------

kmeans_model$cost

# COMMAND ----------

kmeans_model$dataset

# COMMAND ----------

kmeans_model$formula

# COMMAND ----------

kmeans_model$model

# COMMAND ----------

kmeans_model$pipeline

# COMMAND ----------

kmeans_model$summary

# COMMAND ----------

# predict the associated class
predicted <- ml_predict(kmeans_model, iris_tbl) %>%
  collect
table(predicted$Species, predicted$prediction)

# COMMAND ----------

library(ggplot2)
# plot cluster membership
ml_predict(kmeans_model) %>%
  collect() %>%
  ggplot(aes(Petal_Length, Petal_Width)) +
  geom_point(aes(Petal_Width, Petal_Length, col = factor(prediction + 1)),
             size = 2, alpha = 0.5) + 
  geom_point(data = kmeans_model$centers, aes(Petal_Width, Petal_Length),
             col = scales::muted(c("red", "green", "blue")),
             pch = 'x', size = 12) +
  scale_color_discrete(name = "Predicted Cluster",
                       labels = paste("Cluster", 1:3)) +
  labs(
    x = "Petal Length",
    y = "Petal Width",
    title = "K-Means Clustering",
    subtitle = "Use Spark.ML to predict cluster membership with the iris dataset."
  )

# COMMAND ----------

#### PCA ######
pca_model <- tbl(sc, "iris") %>%
  select(-Species) %>%
  ml_pca()

# COMMAND ----------

pca_model

# COMMAND ----------

pca_model$explained_variance

# COMMAND ----------

plot(pca_model$explained_variance/sum(pca_model$explained_variance), ylab="Prop. Variance Explained", xlab="Principle Component")
lines(pca_model$explained_variance/sum(pca_model$explained_variance))

# COMMAND ----------

plot(cumsum(pca_model$explained_variance)/sum(pca_model$explained_variance), ylab="Cumulative Prop. Variance Explained", xlab="Principle Component")
lines(cumsum(pca_model$explained_variance)/sum(pca_model$explained_variance))

# COMMAND ----------

summary(pca_model)

# COMMAND ----------

partitions <- iris_tbl %>%
  sdf_partition(training = 0.7, test = 0.3, seed = 1111)

iris_training <- partitions$training
iris_test <- partitions$test

# COMMAND ----------

nb_model <- iris_training %>%
  ml_naive_bayes(Species ~ .)

# COMMAND ----------

str(nb_model)

# COMMAND ----------

summary(nb_model)

# COMMAND ----------

nb_model$formula

# COMMAND ----------

nb_model$response

# COMMAND ----------

nb_model$pi

# COMMAND ----------

nb_model$theta

# COMMAND ----------

pred <- sdf_predict(iris_test, nb_model)

# COMMAND ----------

pred

# COMMAND ----------

pred_res=collect(pred)

# COMMAND ----------

pred_res$features

# COMMAND ----------

pred_res$probability

# COMMAND ----------

pred_res$label

# COMMAND ----------

ml_multiclass_classification_evaluator(pred)
