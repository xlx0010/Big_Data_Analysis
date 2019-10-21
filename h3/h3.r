
library(ggplot2)

library(sparklyr)
library(dplyr)

sc = spark_connect(master = "local")

## 1
data(diamonds)
diamonds_tbl = copy_to(sc, diamonds, overwrite = T)
src_tbls(sc)

## 1/Linear Regression
lr_model = 
    diamonds_tbl %>%
    #mutate(price_class = as.numeric(price>=3932.8)) %>%
    ml_linear_regression(response = "price", 
                        features = c("carat"))

summary(lr_model)

diamonds_tbl %>%
  select(price, carat) %>%
  ggplot(aes(carat, price)) +
    geom_point(aes(carat, price), size = 2, alpha = 0.5) +
    geom_abline(aes(slope = coef(lr_model)[["carat"]],
                    intercept = coef(lr_model)[["(Intercept)"]]),
                color = "red") +
  labs(
    x = "carat",
    y = "price",
    title = "Linear Regression: carat ~ price",
    subtitle = "Use Spark.ML linear regression to predict price as a function of carat."
  )

## 1/Logistic Regression
partitions <- tbl(sc, "diamonds") %>%
    mutate(price_class = as.numeric(price >= 3932.8)) %>%
    sdf_random_split(train = 0.75, test = 0.25, seed = 1099)  

logit_model = partitions$train %>%
    ml_logistic_regression(price_class ~ carat) 

pred = ml_predict(logit_model, partitions$test) %>%  
    collect  

# Plot
ggplot(pred, aes(x = carat, y = prediction)) +   
geom_point() + theme(plot.title = element_text(hjust = 0.5)) +   
coord_fixed(ratio = 1) +   labs(x = "corat", y = "price_class", title = "price_class vs. corat") 

## 2
set.seed(106)
m = matrix(rnorm (600*2) , ncol =2)
m[1:200, 1] = m[1:200, 1] + 3
m[401:600, 2] = m[401:600, 2] - 3
data = data.frame(x = m[,1], y = m[,2])

data_tbl = copy_to(sc, data, overwrite = T)

partitions <- tbl(sc, "data") %>%
  sdf_random_split(train = 0.8, test = 0.2, seed = 106)

kmeans_model_3 <- partitions$train %>%
  ml_kmeans(features=c("x", "y"), k = 3)

predicted_3 <- ml_predict(kmeans_model, partitions$test) %>%
  collect

ml_predict(kmeans_model_3) %>%
  collect() %>%
  ggplot(aes(x, y)) +
  geom_point(aes(x, y, col = factor(prediction + 1)),
             size = 2, alpha = 0.5) + 
  geom_point(data = kmeans_model_3$centers, aes(x, y),
             col = scales::muted(c("red", "green", "blue")),
             pch = 'x', size = 12) +
  scale_color_discrete(name = "Predicted Cluster",
                       labels = paste("Cluster", 1:3)) +
  labs(
    x = "x",
    y = "y",
    title = "K-Means Clustering",
    subtitle = "K = 3"
  )

## 3

data(baseball, package = 'plyr')
baseball_tbl = copy_to(sc, baseball, overwrite = T)


pca_model <- baseball_tbl %>%
  # #select(g, ab, r, h, X2b, X3b, hr, rbi, sb, cs, bb, so) %>%
  # summarise(
  #   g = as.double(g),
  #   ab = as.double(ab),
  #   r = as.double(r)
  # ) %>%
  select(g, ab, r, h, X2b, X3b, hr, bb) %>%
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

# local compute
X_tbl <- baseball_tbl %>%
  select(g, ab, r, h, X2b, X3b, hr, bb) %>%
  collect()
X = as.matrix(X_tbl)
S = t(X) %*% X
eigen_values = eigen(S)$values

sum(eigen_values[1:1])/sum(eigen_values)

## 4

train = data.frame(age=c(0,0,1,2,2,2,1,0,0,2,0,1,1),
                      income=c(3,3,3,2,1,1,1,2,1,2,2,2,3),
                      credit=c(0,1,0,0,0,1,1,0,0,0,1,1,0),
                      buyer=c(0,0,1,1,1,0,1,0,1,1,1,1,1))
test <- data.frame(age=c(0),
                   income=c(2),
                   credit=c(0))
train_tbl = copy_to(sc, train, overwrite = T)
test_tbl = copy_to(sc, test, overwrite = T)

nb_model <- train_tbl %>% 
  ml_naive_bayes(buyer ~ .)

pred <- ml_predict(nb_model, test_tbl)
#ml_multiclass_classification_evaluator(pred)

pred_res=collect(pred)
pred_res$probability



