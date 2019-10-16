# Databricks notebook source
#Linear Regression

# COMMAND ----------

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

src_tbls(sc) ## code to return all the data frames associated with sc

# COMMAND ----------

iris_tbl %>% count

# COMMAND ----------

head(iris_tbl)

# COMMAND ----------

iris_tbl %>% 
    mutate(Sepal_Width = ROUND(Sepal_Width * 2) / 2) %>% 
    group_by(Species, Sepal_Width) %>%
    summarize(count = n(), Sepal_Length = mean(Sepal_Length), stdev = 
sd(Sepal_Length))

# COMMAND ----------

iris_summary <- 
  iris_tbl %>% mutate(Sepal_Width = ROUND(Sepal_Width * 2) / 2) %>% 
  group_by(Species, Sepal_Width) %>% 
  summarize(count = n(), Sepal_Length = mean(Sepal_Length), stdev = 
sd(Sepal_Length)) %>% 
  collect

# COMMAND ----------

library(ggplot2) 
ggplot(iris_summary, aes(Sepal_Width, Sepal_Length, color = Species)) + 
  geom_line(size = 1.2) + 
  geom_errorbar(aes(ymin = Sepal_Length - stdev, ymax = Sepal_Length +
stdev), width = 0.05) +
  geom_text(aes(label = count), vjust = -0.2, hjust = 1.2, color = "black") +
  theme(legend.position="top")

# COMMAND ----------

## Linear Regression
fit1 <- ml_linear_regression(x = iris_tbl, response = "Sepal_Length", 
                             features = c("Sepal_Width", "Petal_Length", 
"Petal_Width"))
summary(fit1)

# COMMAND ----------

fit1$features

# COMMAND ----------

fit1$response

# COMMAND ----------

fit1$coefficients

# COMMAND ----------

fit1$dataset

# COMMAND ----------

fit1$formula

# COMMAND ----------

fit1$model

# COMMAND ----------

fit1$pipeline

# COMMAND ----------

fit1$pipeline_model

# COMMAND ----------

fit1$summary

# COMMAND ----------

lm_model <- iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  ml_linear_regression(Petal_Length ~ Petal_Width)

# COMMAND ----------

summary(lm_model)

# COMMAND ----------

lm_model$summary

# COMMAND ----------

iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  collect %>%
  ggplot(aes(Petal_Length, Petal_Width)) +
    geom_point(aes(Petal_Width, Petal_Length), size = 2, alpha = 0.5) +
    geom_abline(aes(slope = coef(lm_model)[["Petal_Width"]],
                    intercept = coef(lm_model)[["(Intercept)"]]),
                color = "red") +
  labs(
    x = "Petal Width",
    y = "Petal Length",
    title = "Linear Regression: Petal Length ~ Petal Width",
    subtitle = "Use Spark.ML linear regression to predict petal length as a function of petal width."
  )

# COMMAND ----------

#SDF Partitioning
partitions <- tbl(sc, "iris") %>%
    sdf_partition(training = 0.75, test = 0.25, seed = 1099)  

#fit <- partitions$training %>%
#    ml_linear_regression(Petal_Length ~ Petal_Width) 

# COMMAND ----------

partitions

# COMMAND ----------

fit <- partitions$training %>%
    ml_linear_regression(Petal_Length ~ Petal_Width) 

# COMMAND ----------

fit

# COMMAND ----------

#SDF Partitioning
estimate_mse <- function(df){ 
  
# 'sdf_predict' is deprecated.
ml_predict(fit, df) %>%
    mutate(resid = Petal_Length - prediction) %>%
    summarize(mse = mean(resid ^ 2)) %>%
    
    collect 
}  

sapply(partitions, estimate_mse) 


# COMMAND ----------

#ft string indexing
ft_string2idx <- iris_tbl %>%
    ft_string_indexer("Species", "Species_idx") %>%
    ft_index_to_string("Species_idx", "Species_remap") %>%
    collect  

table(ft_string2idx$Species, ft_string2idx$Species_remap) 

# COMMAND ----------

#SDF_mutate
#ft_string2idx_2 <- iris_tbl %>%
#    mutate(Species_idx = ft_string_indexer(Species)) %>%
#    mutate(Species_remap = ft_index_to_string(Species_idx)) %>%
#    collect    

#ft_string2idx_2 %>%
#    select(Species, Species_idx, Species_remap) %>%
#    distinct 

# COMMAND ----------

# Load dlyr package.
#if (!require("SparkR")) {
#  install.packages("SparkR")
#}
#library(SparkR)

# COMMAND ----------

# Work Flow
mtcars_tbl <- copy_to(sc, mtcars, "mtcars") 

# COMMAND ----------

head(mtcars_tbl)

# COMMAND ----------

str(mtcars_tbl)

# COMMAND ----------

# transform our data set, and then partition into 'training', 'test’ 
mtcars_partitions <- mtcars_tbl %>%
#    sdf_mutate(cyl8 = ft_bucketizer(cyl, c(0,8,12))) %>%
    ft_bucketizer(input_col = "cyl", output_col = "cyl8", splits=c(0,8,12)) %>%
    sdf_partition(training = 0.5, test = 0.5, seed = 888)  

# COMMAND ----------

# fit a linear mdoel to the training dataset 
fit <- mtcars_partitions$training %>%   
    ml_linear_regression(mpg ~ wt + cyl) 

# COMMAND ----------

# summarize the model 
summary(fit)

# COMMAND ----------

# Score the data 
pred <- ml_predict(fit, mtcars_partitions$test) %>%   
     collect  

# COMMAND ----------

# Load dlyr package.
library(ggplot2)

# COMMAND ----------

# Plot the predicted versus actual mpg 
ggplot(pred, aes(x = mpg, y = prediction)) +   
geom_abline(lty = "dashed", col = "red") +   
geom_point() +   theme(plot.title = element_text(hjust = 0.5)) +   
coord_fixed(ratio = 1) +   labs(     x = "Actual Fuel Consumption",     y = "Predicted Fuel Consumption",     title = "Predicted vs. Actual Fuel Consumption"   ) 

# COMMAND ----------

#Logistic Regression

# COMMAND ----------

# Prepare beaver dataset
beaver <- beaver2
beaver$activ <- factor(beaver$activ, labels = c("Non-Active", "Active"))
copy_to(sc, beaver, "beaver")

# COMMAND ----------

# Logistic Regression
beaver_tbl <- tbl(sc, "beaver")

logit_model <- beaver_tbl %>%
  mutate(binary_response = as.numeric(activ == "Active")) %>%
  ml_logistic_regression(binary_response ~ temp)

# COMMAND ----------

summary(logit_model)

# COMMAND ----------

logit_model$summary

# COMMAND ----------

# transform our data set, and then partition into 'training', 'test’ 
beaver_partitions <- beaver_tbl %>%
  mutate(binary_response = as.numeric(activ == "Active")) %>%
    sdf_partition(training = 0.5, test = 0.5, seed = 888)  

# fit a logistics linear mdoel to the training dataset 
logit_model_partitions <- beaver_partitions$training %>%
  ml_logistic_regression(binary_response ~ temp)

# COMMAND ----------

logit_model_partitions$coefficients

# COMMAND ----------

# Score the data 
beaver_pred <- ml_predict(logit_model_partitions, beaver_partitions$test) %>%  collect  

# COMMAND ----------

beaver_pred

# COMMAND ----------

# Plot the active or not versus Temperature
ggplot(beaver_pred, aes(x = temp, y = prediction)) +   
geom_point() +   theme(plot.title = element_text(hjust = 0.5)) +   
coord_fixed(ratio = 1) +   labs(     x = "Temperature",     y = "Active or Not",     title = "Active or Not vs. Temperature"   ) 

# COMMAND ----------

# Genearlized Linear Regression

# COMMAND ----------

mtcars_tbl <- sdf_copy_to(sc, mtcars, name = "mtcars_tbl", overwrite = TRUE) 

# COMMAND ----------

partitions <- mtcars_tbl %>% sdf_partition(training = 0.7, test = 0.3, seed = 1111) 

# COMMAND ----------

mtcars_training <- partitions$training 
mtcars_test <- partitions$test 

# COMMAND ----------

family <- c("gaussian", "gamma", "poisson") 
link <- c("identity", "log") 

family_link <- expand.grid(family = family, link = link, stringsAsFactors = FALSE) 
family_link <- data.frame(family_link, rmse = 0) 

# COMMAND ----------

for(i in 1:nrow(family_link)){ 
  glm_model <- mtcars_training %>% 
  ml_generalized_linear_regression(mpg ~ ., family = family_link[i, 1], link = family_link[i, 2]) 
  pred <- sdf_predict(mtcars_test, glm_model) 
  family_link[i,3] <- ml_regression_evaluator(pred, label_col = "mpg") 
} 

family_link
