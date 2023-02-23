#IA02 Q01: Get dataset ready

# WRITE YOUR CODE HERE
# 1. Load the libraries glmnet, and splines.
library(glmnet)
library(splines)

# 2. Clear all environmental variables.
rm(list=ls())

# 3. Set seed to 5082.
set.seed(5082)

# 4. Read in the data file zillow2223.csv as a dataframe.
#i. Name it my.df.
#ii. Set header to TRUE.
#iii. Use comma as the separator.
#iv. Use the first column “name” as row names (indices).
#v. Read qualitative variables as factors by setting the stringsAsFactors parameter.
my.df <- read.csv("zillow2223.csv", header=TRUE, row.names = 1,  stringsAsFactors = TRUE, sep=",")

# 5. Make zipcode a factor. The model will run incorrectly, if zipcode is still integer.
my.df$zipcode<-as.factor(my.df$zipcode)

# 6. Calculate the age of the house as a new feature. Name it “age”, and add the age feature to the my.df dataframe.
my.df$age <- 2023-my.df$year

# 7. Print summary() of the age feature (#Q01-1)
summary(my.df$age)

# 8. Remove columns we don’t need: section, team, international, masonyear, zestimate, assessment, taxes, willingtopay, year.
my.df <- subset(my.df, select= -c(section, team, international, masonyear, zestimate, assessment, taxes, willingtopay, year))
my.df <- na.omit(my.df)
# 9. Attach my.df so it becomes the default data frame for this project.
attach(my.df)

# 10. Create the x matrix and y vector to be used with glmnet(). Note that glmnet() expects x created by the model.matrix() function.
# i. Create the x matrix using all features except for price. Remove the intercept column from the x matrix.
# ii. Create the y vector using the price variable.

x=model.matrix(price~., data = my.df[,-1])
y=my.df$price

# 11. Create the train (80%) and test (20%) sets following the textbook approach:
# i. Use the sample() function to select row indices to include in the training set. Name this vector trainIndex.
# ii. Use trainIndex to create the training and test dataset.
# iii. Name the training set train.x and train.y.
# iv. Name the test set test.x and test.y.  

trainIndex <- sample(1: nrow(my.df), nrow(my.df)*0.8)
train.x <- x[trainIndex,]
train.y <- y[trainIndex]
test.x <- x[-trainIndex,]
test.y <- y[-trainIndex]

# 12. Print the values of trainIndex. (#Q1-2)
print(trainIndex)

# 13. Print the summary() of the test.y vector. (#Q01-3)
summary(test.y)

##################################

#IA02 Q2: Lasso
# WRITE YOUR CODE HERE
# 1. Include all of the code from the previous question. Do NOT comment out print statements.
# 2. Create a grid vector of 150 elements ranging from 10^4 to 10^-2. We will use this vector to tune the lambda hyperparameter.

grid <- 10^seq(4, -2, length=150)


# 3. Using the glmnet() function, create a lasso model named mod.lasso to predict training y vector using all features from the training x matrix and the grid of lambda values created above.
mod.lasso <- glmnet(train.x, # x matrix
                    train.y, # y vector - we do NOT use the dataframe
                    alpha=1, # 0 is ridge, 1 is lasso
                    lambda=grid)


# 4. Evaluate training model performance with cross-validation. Using the cv.glmnet() function and the same parameters specified above in the creation of mod.lasso (i.e. including the lambda grid vector), create a 12-fold cross-validation model named cv.out.lasso.
cv.out.lasso <- cv.glmnet(train.x, # x matrix
                          train.y, # y vector - we do NOT use the dataframe
                          alpha=1,# 0 is ridge, 1 is lasso
                          nfolds = 12,
                          lambda=grid)


# 5. Print the best cross-validated lambda value (i.e., the one that produces the lowest deviance - do not use the 1-standard error rule here). (#Q2-1)
bestlam <- cv.out.lasso$lambda.min
print(bestlam)

# 6. Make predictions using the model mod.lasso, and the best lambda from the last step. Print a vector of test set predictions. The output should be a table with two columns: the first is a list of student names, and the second is a list of their predictions. (Q02-2)
lasso.pred <- predict(mod.lasso, s=bestlam, newx=test.x)
print(lasso.pred)

# 7. Compute and print the test MSE. (#Q2-3)
mse <- mean((lasso.pred-test.y)^2)
print(mse) #Q08-2

# 8. Print beta coefficients of the model associated with the best lambda. (#Q02-4)
coef(cv.out.lasso, s=bestlam)

# 9. Print.J.T.B’s residual (calculated as Y - Y_hat) (#Q02-5)
jtb.pred <- predict(mod.lasso, s=bestlam, newx=train.x)
print(my.df$price[row.names(my.df)=="J.T.B"] - jtb.pred[row.names(jtb.pred)=="J.T.B"])

##############################################################

#IA02 Q3: Ridge for Classification
# WRITE YOUR CODE HERE

# 1. Include all of the code from the previous questions. Do NOT comment out print statements.
# 2. Set seed to 5082.
set.seed(5082)

# 3. Create the x matrix using all features except for zipcode, and the y vector using the zipcode feature. These are what glmnet need.
x=model.matrix(zipcode~., data = my.df[,-1])
y=my.df$zipcode

# 4. Create the train (80%) and test (20%) data sets using the sample() function, following the textbook approach.

trainIndex <- sample(1: nrow(my.df), nrow(my.df)*0.8)
train.x <- x[trainIndex,]
train.y <- y[trainIndex]
test.x <- x[-trainIndex,]
test.y <- y[-trainIndex]

# 5. Using the glmnet() function, create a ridge model named mod.ridge that predicts the training y vector using the training x matrix and the grid of lambda values created on the last page.
mod.ridge <- glmnet(train.x, # x matrix
                    train.y, # y vector - we do NOT use the dataframe
                    alpha=0, # 0 is ridge, 1 is lasso
                    lambda=grid,
                    family = "binomial")

# 6. Evaluate training model performance with cross-validation. Using the cv.glmnet() function and the same parameters specified above in the creation of mod.ridge (i.e. including the lambda grid vector), create a 12-fold cross-validation model named cv.out.ridge.
cv.out.ridge <- cv.glmnet(train.x, # x matrix
                          train.y, # y vector - we do NOT use the dataframe
                          alpha=0,# 0 is ridge, 1 is lasso
                          nfolds = 12,
                          lambda=grid,
                          family = "binomial")

# 7. Print the best lambda value from cv.out.ridge (#Q03-1)
bestlam.ridge<-cv.out.ridge$lambda.min
print(bestlam.ridge) #Q15-1

# 8. Print a vector of test set predictions, using the best lambda and the mod.ridge model, create. Set type to “response.” The output should be a table with two columns: the first is a list of student names, and the second is a list of their predictions. (#Q03-2)
pred.ridge <- predict(mod.ridge, s=bestlam.ridge, newx=test.x, type="response")
print(pred.ridge)

# 9. Print beta coefficients of the model associated with the best lambda. (#Q03-3)
coef(mod.ridge, s=bestlam.ridge)

# 10. Print the test set predictions in terms of zipcode (i.e., 23185, or 23188 - no quotation marks). Use contrasts() to determine which zipcode is the positive class. Use >= .5 as the classification rule. The output should be a table with two columns: the first is a list of student names, and the second is a list of zipcodes. (#Q03-4)
contrasts(train.y)
class.pred <- ifelse(pred.ridge >= 0.5, 23188, 23185)

# 11. Print a list of TRUE/FALSE values indicating whether each test prediction is correct (TRUE) or not (FALSE). The output should be a table with two columns: the first is a list of student names, and the second is a list of TRUE/FALSE values. (#Q03-5)
print(ifelse(class.pred==test.y,T,F))

# 12. Print overall model accuracy rate of the test set predictions (with 3 decimal values; not percentage %). (#Q03-6)
print(mean(class.pred==test.y))




#################################################

#IA02 Q04
# WRITE YOUR CODE HERE
# 1. Include all of the code from the previous questions. Do NOT comment out print statements.
# 2. Set seed to 5082.
set.seed(5082)

# 3. Create a vector of 8 elements, filled with zeros. Name this vector ns.cv.mse. We will use it to store cross-validation MSEs.
#ns.cv.mse <- rep(0,8)
test.y <- my.df$price[-trainIndex]
ns.cv.mse=seq(from=0, to=0, length.out=8)



# 4. Train a cubic Natural Splines model to predict price, and evaluate its cross-validation test performance.
  #i. Use sqft as a predictor, with the training dataset.
  #ii. Compare test MSEs of 1 to 8 knots, with a for loop, and the test dataset. Be sure to use price, not zipcode as the target in test.
  #iii. Store the cross-validation errors into ns.cv.mse.
#natural splines

for (i in 1:8){
  NSpline.model <- glm(price~ns(sqft,df=3+i-2),data=my.df[trainIndex,])
  NSpline.pred <- predict(NSpline.model, newdata = my.df[-trainIndex,])
  ns.cv.mse[i] <- mean((NSpline.pred - test.y)^2)
  }

# 5. Print ns.cv.mse. (#Q04-1)
print(ns.cv.mse)

# 6. In the RStudio window, plot the cross-validation mses by number of knots. Choose the best model using the 1-SD rule, based on the plot.

plot(ns.cv.mse[1:8])

(stdev <- sd(ns.cv.mse[1:8]))
(min <- which.min(ns.cv.mse))
abline(h=ns.cv.mse[min] + stdev, col = "red", lty = "dashed")
best.knots <- 5


# 7. Print price predictions using the best natural splines model built with the best lambda value from above. (#Q04-2)

BestNSModel <- lm(price~ns(sqft, df=3+ best.knots -2), data=my.df[trainIndex,])
price_hat_ns <- predict(BestNSModel, newdata=my.df[-trainIndex,])
price_hat_ns #Q04-2
