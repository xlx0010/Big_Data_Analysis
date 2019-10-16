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

partitions <- iris_tbl %>%
  sdf_partition(training = 0.7, test = 0.3, seed = 1111)

iris_training <- partitions$training
iris_test <- partitions$test

# COMMAND ----------

dt_model <- iris_training %>% 
ml_decision_tree(Species ~ .) 

# COMMAND ----------

dt_model$model$num_nodes

# COMMAND ----------

dt_model

# COMMAND ----------

dt_model$features

# COMMAND ----------

dt_model$model$depth

# COMMAND ----------

dt_model$model$feature_importances

# COMMAND ----------

dt_model$model$num_nodes

# COMMAND ----------

dt_model$model$thresholds

# COMMAND ----------

dt_model$model$param_map

# COMMAND ----------

pred <- sdf_predict(iris_test, dt_model) 
ml_multiclass_classification_evaluator(pred)

# COMMAND ----------

rf_model <- iris_training %>% ml_random_forest(Species ~ ., type = "classification")

# COMMAND ----------

# predict the associated class
pred <- sdf_predict(iris_test, rf_model)
ml_multiclass_classification_evaluator(pred)

# COMMAND ----------

gbt_model <- iris_training %>% ml_gradient_boosted_trees(Sepal_Length ~ Petal_Length +
Petal_Width)

# COMMAND ----------

# predict the associated class
pred <- sdf_predict(iris_test, gbt_model)
ml_regression_evaluator(pred, label_col = "Sepal_Length")

# COMMAND ----------

library(ggplot2)
# plot cluster membership
sdf_predict(iris_test, dt_model)  %>%
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
