options(warn=-1)
library(data.table)
library(entropy)
library(ggplot2)
library(caTools)
library(ROCR)
library(rpart)
library(e1071)
library(rpart)
library(rpart.plot)
library(caret)
library(corrplot)
library(pROC)

#loading data
attrition_dat = fread("C:/Users/admin/Documents/Attrition.csv")

# Checking the dimension of data
dim(attrition_dat)
head(attrition_dat)

## check null values 
sum(is.na(attrition_dat))

# Checking the classes of columns in overall dataset
# unlist(lapply(attrition_dat, class))
str(attrition_dat)

## converting categorical columns to factors
cat_cols = c("Attrition","BusinessTravel","Department","Education","EducationField","EnvironmentSatisfaction","Gender",
             "JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","Over18","OverTime",
             "PerformanceRating","RelationshipSatisfaction","StandardHours","StockOptionLevel","WorkLifeBalance")

attrition_dat[,cat_cols] <-  lapply(attrition_dat[,cat_cols,with=FALSE], as.factor)

# Checking summary of data to know about the distribution of data
summary(attrition_dat)

# List of continuous variables in data
cont_vars <- c("Age", "DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome","MonthlyRate",
               "NumCompaniesWorked","PercentSalaryHike","TotalWorkingYears","TrainingTimesLastYear",
               "YearsAtCompany", "YearsInCurrentRole",  "YearsSinceLastPromotion","YearsWithCurrManager",
               "EmployeeCount")
View(cont_vars)

# Distribution of continious variables across attrition
melt_attrition_dat = melt(attrition_dat[,c("Attrition", cont_vars),with=FALSE], id.var = "Attrition")
p <- ggplot(data = melt_attrition_dat , aes(x=variable, y=value)) + geom_boxplot(aes(fill= Attrition))
p <- p + facet_wrap( ~ variable, scales="free")
p


### % attrition across categorical variables
freq_tbl <-  apply(attrition_dat[,cat_cols,with=FALSE],2, function(x) table(attrition_dat$Attrition,x))
freq_tbl <- lapply(freq_tbl,function(x) as.data.frame.matrix(x))

perc_attrition_plot <- list()
i =0
for(name in names(freq_tbl)[-1]){
  i <- i +1
  var_data <- data.frame(apply(freq_tbl[name][[1]],2, function(x) x[2]/sum(x)))
  colnames(var_data) <- name
  my_plot <- ggplot(data=var_data, aes(x=row.names(var_data), y=var_data[,name])) +  geom_bar(stat="identity",fill='red') +
    ylim(0.0,1.0) + ylab("%attrition") + xlab(name) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
  plot(my_plot)
  remove(my_plot)
}


### % Hike vs salary color by attrition
ggplot(data = attrition_dat,aes(x = MonthlyIncome,y = YearsSinceLastPromotion,color=attrition_dat$Attrition )) + geom_point()

# Making correlation matrix for numeric variables
corrmat_cont_vars <- cor(attrition_dat[,setdiff(cont_vars,"EmployeeCount"),with=FALSE])
View(corrmat_cont_vars)

# Plotting correlation plot
corrplot(corrmat_cont_vars)

## Feature Selection ##
## Pre-modeling feature selection (variance and entropy based) 
normalised_entropy_cat_vars = unlist(lapply(attrition_dat[,cat_cols,with=FALSE],function(x) entropy(table(x)/length(unique(x[!is.na(x)])))))/unlist(lapply(attrition_dat[,cat_cols,with=FALSE], function(x) log2(length(x[!is.na(x)]))))
low_entropy_variables = names(normalised_entropy_cat_vars[normalised_entropy_cat_vars ==0])

# normalised_attrition_data <- attrition_dat[,cont_vars,with=FALSE]
variance_cont_vars <- apply(attrition_dat[,cont_vars,with=FALSE],2, function(x) (x-min(x))/(max(x)-min(x)))
variance_cont_vars <- as.data.frame.matrix(variance_cont_vars)
variance_cont_vars <- apply(variance_cont_vars,2,var)
low_variance_vars <- names(variance_cont_vars[is.na(variance_cont_vars)==TRUE ])

print(paste(c("Variables with low entropy: ",low_entropy_variables),collapse = " "))
print(paste(c("Variables with low variance: ",low_variance_vars),collapse = " "))

## Removing variables with low variance and low entropy
attrition_dat1 <- attrition_dat[,-c(low_entropy_variables,low_variance_vars,"EmployeeNumber"),with=FALSE]
dim(attrition_dat1)

## Normalising contionious variables
attrition_dat1[,cont_vars[!(cont_vars %in% low_variance_vars)]] <- as.data.frame.matrix(apply(attrition_dat[,cont_vars[!(cont_vars %in% low_variance_vars)],with=FALSE],2, function(x) (x-min(x))/(max(x)-min(x))))
dim(attrition_dat1)

## Checking attrition table
"Attrition"
table(attrition_dat1$Attrition)

##Spliting data into training and testing using Stratified sampling
set.seed(90)
split = sample.split(attrition_dat1$Attrition,SplitRatio = 0.75)
attrition_train <- subset(attrition_dat1,split == TRUE)
attrition_test <- subset(attrition_dat1,split == FALSE)

print(c("Row in Train",nrow(attrition_train)))
print(c("Row in Test", nrow(attrition_test)))

##Distribution of churn in Train
table(attrition_train$Attrition)

##Distribution of churn in test
table(attrition_test$Attrition)


## Model Building ##
##Logistic regression
set.seed(101)
attr_log <- glm(Attrition ~ ., data = attrition_train,family = 'binomial')
summary(attr_log)

### Predicting for test data
predict_test = predict(attr_log,newdata = attrition_test,type = 'response')
View(predict_test)

##Checking various thresholds( 0.5,0.7.0.1) and analyzing the results
##Threshold - 0.5
print("Confusion matrix for threshold 0.5")

thershold= 0.5

confusion_mat <- table(attrition_test$Attrition, predict_test > thershold)
confusion_mat
# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- confusion_mat[4]
tp_plus_fn <- confusion_mat[4] + confusion_mat[2]

sensitivity <- tp/tp_plus_fn
print(c("sensitivity",sensitivity))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tn_plus_fp <- confusion_mat[1] + confusion_mat[3]

specificity <- tn/tn_plus_fp
print(c("specificity",specificity))

## Threshold - 0.7 
print("Confusion matrix for threshold 0.7")

thershold= 0.7

confusion_mat <- table(attrition_test$Attrition, predict_test > thershold)
confusion_mat
# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- confusion_mat[4]
tp_plus_fn <- confusion_mat[4] + confusion_mat[2]

sensitivity <- tp/tp_plus_fn
print(c("sensitivity",sensitivity))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tn_plus_fp <- confusion_mat[1] + confusion_mat[3]

specificity <- tn/tn_plus_fp
print(c("specificity",specificity))



## Threshold - 0.1
print("Confusion matrix for threshold 0.1")

thershold= 0.1
confusion_mat <- table(attrition_test$Attrition, predict_test > thershold)
confusion_mat

# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- confusion_mat[4]
tp_plus_fn <- confusion_mat[4] + confusion_mat[2]

sensitivity <- tp/tp_plus_fn
print(c("sensitivity",sensitivity))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tnplusFP <- confusion_mat[1] + confusion_mat[3]

specificity <- tn/tnplusFP
print(c("specificity",specificity))


## Plotting Receiver operator characteristics curve to decide better on threshold ##
rocr_pred_logistic_best_treshold = prediction(predict_test ,attrition_test$Attrition)
rocr_perf_logistic_best_treshold = performance(rocr_pred_logistic_best_treshold,'tpr','fpr')
plot(rocr_perf_logistic_best_treshold,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))

##From plot we observed 0.3 is the best threshold and will evaluate model performance with this threshold
thershold_best_log = 0.3

conf_mat_logistic_best_treshold <- table(attrition_test$Attrition ,predict_test > thershold_best_log)
View(conf_mat_logistic_best_treshold)

#checking accuracy
accuracy_logistic_best_treshold <- (conf_mat_logistic_best_treshold[1] + conf_mat_logistic_best_treshold[4])/(conf_mat_logistic_best_treshold[1]+conf_mat_logistic_best_treshold[2]+conf_mat_logistic_best_treshold[3]+conf_mat_logistic_best_treshold[4])
"Confusion matrix for best threshold (logistic regression)"
conf_mat_logistic_best_treshold
"Model Performance"
print(c("Accuracy",accuracy_logistic_best_treshold))

# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- conf_mat_logistic_best_treshold[4]
tp_plus_fn <- conf_mat_logistic_best_treshold[4] + conf_mat_logistic_best_treshold[2]

sensitivity_logistic_best_treshold <- tp/tp_plus_fn
print(c("sensitivity",sensitivity_logistic_best_treshold))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tn_plus_fp <- conf_mat_logistic_best_treshold[1] + conf_mat_logistic_best_treshold[3]

specificity_logistic_best_treshold <- tn/tn_plus_fp
print(c("specificity",specificity_logistic_best_treshold))


## Checking factors with same levels in data & Reference
str(predict_test > thershold_best_log)
levels(predict_test > thershold_best_log)

str(attrition_test$Attrition)
levels(attrition_test$Attrition)

View(predict_test > thershold_best_log)
View(attrition_test$Attrition)

##checking variables which have p-value less than 0.05 that are considered important 
##reducing complexity of the model by selecting important features
summary_coeff_pval =  as.data.frame.matrix(summary(attr_log)$coef)
summary_coeff_pval[summary_coeff_pval$`Pr(>|z|)` <= 0.05,]

##Built model with important variables only and evaluate the trade off between complexity and accuracy

important_vars_logistic <- c('Age','BusinessTravel','DistanceFromHome','EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel',
                             'JobSatisfaction','NumCompaniesWorked','OverTime','RelationshipSatisfaction',
                             'StockOptionLevel','WorkLifeBalance','YearsSinceLastPromotion', 'YearsWithCurrManager')

##Model
set.seed(2001)
attr_log_imp_vars <- glm(Attrition ~ ., data = attrition_train[,c('Attrition' ,important_vars_logistic),with=FALSE],family = 'binomial')
summary(attr_log_imp_vars)

#predict
predict_test_imp_log <- predict(attr_log_imp_vars,newdata = attrition_test[,c('Attrition' ,important_vars_logistic),with=FALSE],type = 'response')
View(predict_test_imp_log)

#Building plot
rocr_pred_log_imp_vars = prediction(predict_test_imp_log ,attrition_test$Attrition)
rocr_perf_log_imp_vars = performance(rocr_pred_log_imp_vars,'tpr','fpr')
plot(rocr_perf_log_imp_vars,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))

##From the above plot we observe that 0.3 is the best threshold. Now we will evaluate model performance witn threshold 0.3
threshold_log_imp_vars = 0.3
conf_mat_log_imp_vars <- table(attrition_test$Attrition ,predict_test_imp_log > threshold_log_imp_vars)

"Confusion matrix model with only important variable"
conf_mat_log_imp_vars
# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- conf_mat_log_imp_vars[4]
tp_plus_fn <- conf_mat_log_imp_vars[4] + conf_mat_log_imp_vars[2]

sensitivity_log_imp_vars <- tp/tp_plus_fn
print(c("sensitivity",sensitivity_log_imp_vars))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- conf_mat_log_imp_vars[1]
tn_plus_fp <- conf_mat_log_imp_vars[1] + conf_mat_log_imp_vars[3]

specificity_log_imp_vars <- tn/tn_plus_fp
print(c("specificity",specificity_log_imp_vars))

#Predicting
# accuracy
accuracy_log_imp_vars <- (conf_mat_log_imp_vars[1] + conf_mat_log_imp_vars[4])/(conf_mat_log_imp_vars[1]+conf_mat_log_imp_vars[2]+conf_mat_log_imp_vars[3]+conf_mat_log_imp_vars[4])
print(c("Accuracy" ,accuracy_log_imp_vars))


##Classification and Regression Trees (CART) ##

set.seed(30001)
cart <- rpart(Attrition ~ ., method="class", data=attrition_train,)

# Predicting for test data
predict_test_cart = as.data.frame.matrix(predict(cart,newdata = attrition_test,type = "prob"))
View(predict_test_cart)

predict_test_cart = predict_test_cart$Yes

##Plotting 
rocr_pred_cart = prediction(predict_test_cart ,attrition_test$Attrition)
rocr_perf_cart = performance(rocr_pred_cart,'tpr','fpr')
plot(rocr_perf_cart,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))

## threshold 0.4 is the best threshold in above cart technique

threshold_cart <- 0.4
conf_mat_cart <- table(attrition_test$Attrition ,predict_test_cart > threshold_cart)


# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- conf_mat_cart[4]
tp_plus_fn <- conf_mat_cart[4] + conf_mat_cart[2]

sensitivity_cart <- tp/tp_plus_fn
"sensitivity"
sensitivity_cart

# specificity tnr--> specificity = tn/(tn+FP)
tn <- conf_mat_cart[1]
tn_plus_fp <- conf_mat_cart[1] + conf_mat_cart[3]

specificity_cart <- tn/tn_plus_fp
"specificity"
specificity_cart


#Predicting
# accuracy
accuracy_cart <- (conf_mat_cart[1] + conf_mat_cart[4])/(conf_mat_cart[1]+conf_mat_cart[2]+conf_mat_cart[3]+conf_mat_cart[4])
"Confusion matrix"
conf_mat_cart
"Accuracy of model"
accuracy_cart

# checking Top 10 important variables with CART 
View(cart)
sort(cart$variable.importance,decreasing = TRUE)[1:10]

cart_imp_vars <- names(sort(cart$variable.importance,decreasing = TRUE)[1:10])
View(cart_imp_vars)

## build MOdel for top 10 imp var 
cart_model_imp_vars <- rpart(Attrition ~ ., method="class", data=attrition_train[,c("Attrition",cart_imp_vars),with= FALSE])

# summary(cart)
predict_test_cart_model_imp_vars = as.data.frame.matrix(predict(cart_model_imp_vars,newdata = attrition_test[,c("Attrition",cart_imp_vars),with= FALSE],type = "prob"))

predict_test_cart_model_imp_vars = predict_test_cart_model_imp_vars$Yes
View(predict_test_cart_model_imp_vars)

##plotting
rocr_pred_cart_model_imp_vars = prediction(predict_test_cart_model_imp_vars ,attrition_test$Attrition)
rocr_perf_cart_model_imp_vars = performance(rocr_pred_cart_model_imp_vars,'tpr','fpr')
# plot(rocr_perf)
# plot(rocr_perf,colorize=TRUE)
plot(rocr_perf_cart_model_imp_vars,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))


##building confusion matrix for cart model
threshold_cart_model_imp_vars <- 0.3
conf_mat_cart_model_imp_vars <- table(attrition_test$Attrition ,predict_test_cart_model_imp_vars > threshold_cart_model_imp_vars)


# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- conf_mat_cart_model_imp_vars[4]
tp_plus_fn <- conf_mat_cart_model_imp_vars[4] + conf_mat_cart_model_imp_vars[2]

sensitivity_cart_model_imp_vars <- tp/tp_plus_fn
"sensitivity"
sensitivity_cart_model_imp_vars

# specificity tnR--> specificity = tn/(tn+FP)
tn <- conf_mat_cart_model_imp_vars[1]
tn_plus_fp <- conf_mat_cart_model_imp_vars[1] + conf_mat_cart_model_imp_vars[3]

specificity_cart_model_imp_vars <- tn/tn_plus_fp
"specificity"
specificity_cart_model_imp_vars


#Predicting
# accuracy
accuracy_cart_model_imp_vars <- (conf_mat_cart_model_imp_vars[1] + conf_mat_cart_model_imp_vars[4])/(conf_mat_cart_model_imp_vars[1]+conf_mat_cart_model_imp_vars[2]+conf_mat_cart_model_imp_vars[3]+conf_mat_cart_model_imp_vars[4])
"Confusion matrix"
conf_mat_cart_model_imp_vars
"Accuracy of model"
accuracy_cart_model_imp_vars


## Plotting tree to check best fit
prp(cart_model_imp_vars)

##Feature engineering ##
##to improve the model by engineering new features
attrition_train_new <- attrition_train
attrition_test_new <- attrition_test
View(attrition_test_new)
View(attrition_train_new)

is_first_company = attrition_dat$NumCompaniesWorked == 1
View(is_first_company)

loyalty =  attrition_dat$YearsAtCompany/attrition_dat$TotalWorkingYears
View(loyalty)

## lower the number more volatile is the employee
volatility = attrition_dat$TotalWorkingYears/attrition_dat$NumCompaniesWorked
volatility[which(is.infinite(volatility))] <- attrition_dat$TotalWorkingYears[which(is.infinite(volatility))]
View(volatility)

attrition_train_new$IsFirstCompany <- is_first_company[split== TRUE]
attrition_test_new$IsFirstCompany <- is_first_company[split== FALSE]

attrition_train_new$Loyalty <- loyalty[split== TRUE]
attrition_test_new$Loyalty <- loyalty[split== FALSE]

attrition_train_new$Volatility <- volatility[split== TRUE]
attrition_test_new$Volatility <- volatility[split== FALSE]

new_features = c('IsFirstCompany','Loyalty','Volatility')
View(new_features)

### creating model with selected important features & new features
attr_log_imp_vars_new_ftr <- glm(Attrition ~ ., data = attrition_train_new[,c('Attrition' ,important_vars_logistic,new_features),with=FALSE],family = 'binomial')
summary(attr_log_imp_vars_new_ftr)

predict_test_imp_log_new_ftr <- predict(attr_log_imp_vars_new_ftr,newdata = attrition_test_new[,c('Attrition' ,important_vars_logistic,new_features),with=FALSE],type = 'response')
View(predict_test_imp_log_new_ftr)


rocr_pred_log_imp_vars_new_ftr = prediction(predict_test_imp_log_new_ftr ,attrition_test_new$Attrition)
rocr_perf_log_imp_vars_new_ftr = performance(rocr_pred_log_imp_vars_new_ftr,'tpr','fpr')
plot(rocr_perf_log_imp_vars_new_ftr,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))

### Confusion matrix model with only important variable and new features

threshold_log_imp_vars_new_ftr = 0.3
conf_mat_log_imp_vars_new_ftr <- table(attrition_test_new$Attrition ,predict_test_imp_log_new_ftr > threshold_log_imp_vars_new_ftr)

"Confusion matrix model with only important variable and new features"
conf_mat_log_imp_vars_new_ftr
# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- conf_mat_log_imp_vars_new_ftr[4]
tp_plus_fn <- conf_mat_log_imp_vars_new_ftr[4] + conf_mat_log_imp_vars_new_ftr[2]
sensitivity_log_imp_vars_new_ftr <- tp/tp_plus_fn
print(c("sensitivity",sensitivity_log_imp_vars_new_ftr))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- conf_mat_log_imp_vars_new_ftr[1]
tn_plus_fp <- conf_mat_log_imp_vars_new_ftr[1] + conf_mat_log_imp_vars_new_ftr[3]

specificity_log_imp_vars_new_ftr <- tn/tn_plus_fp
print(c("specificity",specificity_log_imp_vars_new_ftr))

#Predicting
# accuracy
accuracy_log_imp_vars_new_ftr <- (conf_mat_log_imp_vars_new_ftr[1] + conf_mat_log_imp_vars_new_ftr[4])/(conf_mat_log_imp_vars_new_ftr[1]+conf_mat_log_imp_vars_new_ftr[2]+conf_mat_log_imp_vars_new_ftr[3]+conf_mat_log_imp_vars_new_ftr[4])

print(c("Accuracy" ,accuracy_log_imp_vars_new_ftr))


######## MODEL SELECTION ######
data.frame(list("model_name" = c("cart all variables","cart important variables","logistic all variables","logistic important variables","Logistic with feature engineering"),
                "Sensitivity" = c(sensitivity_cart,sensitivity_cart_model_imp_vars,sensitivity_logistic_best_treshold,sensitivity_log_imp_vars,sensitivity_log_imp_vars_new_ftr),
                "Specificity" = c(specificity_cart,specificity_cart_model_imp_vars,specificity_logistic_best_treshold,specificity_log_imp_vars,specificity_log_imp_vars_new_ftr),
                "Accuracy" = c(accuracy_cart,accuracy_cart_model_imp_vars,accuracy_logistic_best_treshold,accuracy_log_imp_vars,accuracy_log_imp_vars_new_ftr)))

## Plot to study the model AUC
plot(roc(attrition_test$Attrition, predict_test_cart), print.auc=TRUE)
plot(roc(attrition_test$Attrition, predict_test_cart_model_imp_vars), print.auc = TRUE,col = "green", print.auc.y = .1, add = TRUE)
plot(roc(attrition_test$Attrition, predict_test_imp_log), print.auc = TRUE,col = "blue", print.auc.y = .2, add = TRUE)
plot(roc(attrition_test$Attrition, predict_test), print.auc = TRUE,col = "red", print.auc.y = .3, add = TRUE)
plot(roc(attrition_test$Attrition, predict_test_imp_log_new_ftr), print.auc = TRUE,col = "pink", print.auc.y = .4, add = TRUE)