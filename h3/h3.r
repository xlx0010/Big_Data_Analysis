
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

summary(lrModel_carat_price)

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
    sdf_random_split(training = 0.75, test = 0.25, seed = 1099)  

## 2




