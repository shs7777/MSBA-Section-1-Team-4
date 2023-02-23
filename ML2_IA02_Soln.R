#IA02 Q01: Get dataset ready

# CODIO SOLUTION BEGIN
#1. load libraries
library(glmnet)
library(splines)

#2. Clear environment
rm(list=ls())

#3. Set seed
set.seed(5082)
#4. Read data file
my.df = read.csv("zillow2223.csv", header=TRUE, sep=',', row.names="name", stringsAsFactors = TRUE)

x <- model.matrix(price~., my.df)[, -1] 
y <- my.df$price
n = nrow(my.df)
trainIndex = sample(1:n, size = n* 0.8)
train.x = x[trainIndex ,]
test.x = x[-trainIndex ,]
train.y = y[trainIndex]
test.y = y[-trainIndex]

#5. Make  zipcode factor. # The model will run incorrectly, if zipcode is still int.
my.df$zipcode <- factor(my.df$zipcode)
#6. Calculate the age of the house variable ("age") and add the age variable to the dataframe
my.df$age <- 2023 - my.df$year
#7. Print summary of age
summary(my.df$age) #Q01-1
#8. Remove features we won't need
my.df<-subset(my.df, select = -c(section,team, international, masonyear, zestimate, assessment, taxes, willingtopay, year))
#9. Attach data frame
attach(my.df)
#10. Create the X matrix using all features except for price, and the Y matrix using the Price variable, that glmnet expects
x <- model.matrix(price~., my.df)[, -1]
y<- my.df$price
#11. Create the train (80%) and test (20%) data sets using the sample() function, following the textbook approach.
n <- nrow(my.df)
trainIndex <- sample(n, .8 * n)
train.x <- x[trainIndex, ]
test.x <- x[-trainIndex, ]
train.y <- my.df$price[trainIndex]
test.y <- my.df$price[-trainIndex]
#12. Print trainIndex values
trainIndex #Q01-2
#13. Print summary() of test.y
summary(test.y) #Q01-3
# CODIO SOLUTION END





#IA02 Q2: Lasso
# CODIO SOLUTION BEGIN
#1. Create a **grid** vector of 150 elements ranging from 10^-2 to 10^4. We will use this vector to tune the **lambda** hyperparameter.
grid = 10 ^ seq(4, -2, length=150)
#2. Using the glmnet() function, create a lasso model named **mod.lasso** to predict training y's using the training x's and the grid of lambda values created above.
mod.lasso <- glmnet(train.x, # X matrix 
                    train.y, # Y vector
                    alpha=1, # use lasso
                    lambda=grid)
#3. Evaluate training model performance using cross-validation. Using the cv.glmnet() function and the same parameters used above in the creation of mod.lasso (i.e. including the lambda grid vector), create a 12-fold cross-validation model named **cv.out.lasso**.
cv.out.lasso <- cv.glmnet(train.x,  
                          train.y, 
                          alpha=1, 
                          lambda=grid,
                          nfolds=12) # 12-fold cross-validation
#4. **Print** the best cross-validated lambda value (the one that produces the lowest deviance - do not use the 1-standard error rule here). **(#Q2-1)**
bestlam.lasso <- cv.out.lasso$lambda.min
bestlam.lasso #Q02-1
#5. Make predictions using the model **mod.lasso**, and the best lambda from the last step. **Print** a vector of test set predictions. **(Q02-2)**
lasso.pred <- predict(mod.lasso, 
                      s=bestlam.lasso, 
                      newx=test.x)
lasso.pred #Q02-2
#6. Compute and **print** the test MSE. **(#Q2-3)**
MSE.lasso <- mean((lasso.pred - test.y)^2)
MSE.lasso #Q02-3
#7. **Print** the coefficients of the model associated with the best lambda. **(#Q2-4)**
lasso.coefficients <- predict(mod.lasso, 
                              s=bestlam.lasso, 
                              type="coefficients")
lasso.coefficients #Q2-4
#8. **Print** J.T.B's residual (calculated as Y - Y_hat) **(#Q2-5)**
my.df["J.T.B", "price"]-lasso.pred["J.T.B",] #Q02-5
# CODIO SOLUTION END



#IA02 Q3: Ridge for Classification
# CODIO SOLUTION BEGIN
#Next, we will use ridge regression to predict **zipcode**.
#1. Set seed to 5082
set.seed(5082)
#2. Create the X matrix using all features except for zipcode, and the Y vector using the zipcode feature, that glmnet expects
x = model.matrix(zipcode~., my.df)[, -1]
y<-my.df$zipcode
#3. Create the train (80%) and test (20%) data sets using the sample() function, following the textbook approach.
n <- nrow(my.df)
trainIndex <- sample(n, .8 * n)
train.x <- x[trainIndex, ]
test.x <- x[-trainIndex, ]
train.y <- my.df$zipcode[trainIndex]
test.y <- my.df$zipcode[-trainIndex]
#4. Using the glmnet() function, create a ridge model named mod.ridge that predicts the training y's using the training x's and the grid of lambda values created above.
mod.ridge <- glmnet(train.x,  
                    train.y, 
                    alpha=0, # ridge
                    lambda=grid,
                    family = binomial) #logistic
#5. Evaluate training model performance using cross-validation. Using the cv.glmnet() function and the same parameters used above in the creation of mod.ridge (i.e. including the lambda grid vector), create a 12-fold cross-validation model named cv.out.ridge
cv.out.ridge <- cv.glmnet(train.x,  
                          train.y, 
                          alpha=0, 
                          lambda=grid,
                          nfolds=12,
                          family = binomial)
#6. **Print** the best lambda value from **cv.out.ridge (#Q03-1)**
bestlam.ridge <- cv.out.ridge$lambda.min
bestlam.ridge #Q03-1
#7. Using the best lambda and the model named mod.lasso, create a vector of test set predictions.
ridge.pred <- predict(mod.ridge, 
                      s=bestlam.ridge, 
                      newx=test.x, 
                      type = "response"
)
ridge.pred #Q03-2

#8. Print coefficients
ridge.coef <- predict(mod.ridge, 
                      s=bestlam.ridge, 
                      newx=test.x, 
                      type = "coefficients"
)
ridge.coef #Q03-3
#9. Print test set predictions in terms of 23188 or 23185.
zipcode_hat <-ifelse(ridge.pred>=.5, 23188, 23185)
zipcode_hat #Q03-4
#10. **Print** a list of TRUE/FALSE values indicating whether each test prediction is correct (TRUE) or not (FALSE). **(#Q03-5)**
test.y == zipcode_hat #Q03-5
#11. **Print** accuracy rate of the test set predictions. **(#Q03-6)**
mean(test.y == zipcode_hat) #Q03-6
# CODIO SOLUTION END





#IA02 Q04
# CODIO SOLUTION BEGIN

#2. Set seed to 5082
set.seed(5082)

y <- my.df$price
n = nrow(my.df)
trainIndex = sample(1:n, size = n* 0.8)
train.x = x[trainIndex ,]
test.x = x[-trainIndex ,]
train.y = y[trainIndex]
test.y = y[-trainIndex]

#3. Create a vector of 8 elements, filled with zeros
ns.cv.mse=seq(from=0, to=0, length.out=8)

#4. Train a cubic Natural Splines model to predict price, and evaluate its cross-validation test performance.
#i. Use sqft as a predictor, with the training dataset.
#ii. Compare test MSEs of 1 to 8 knots, with a for loop, and the test dataset. Be sure to use price, not zipcode as the target in test.
#iii. Store the cross-validation errors into ns.cv.mse.
#natural splines

for(num.knots in 1:8){
  NSmodel <- lm(price~ns(sqft, df=3+ num.knots -2), data=my.df[trainIndex,])
  ns.pred <- predict(NSmodel, newdata=my.df[-trainIndex,])
  ns.cv.mse[num.knots] <- mean((ns.pred - test.y)^2)
}
#5 Print ns.cv.mse.
ns.cv.mse #Q04-1

#6. plot 
plot(ns.cv.mse[1:8])
(stdev <- sd(ns.cv.mse[1:8]))
(min <- which.min(ns.cv.mse))
abline(h=ns.cv.mse[min] + stdev, col = "red", lty = "dashed")
best.knots=6
#7. Print price predictions using best model
BestNSModel <- lm(price~ns(sqft, df=3+ best.knots -2), data=my.df[trainIndex,])
price_hat_ns <- predict(BestNSModel, newdata=my.df[-trainIndex,])
price_hat_ns #Q04-2
# CODIO SOLUTION END
