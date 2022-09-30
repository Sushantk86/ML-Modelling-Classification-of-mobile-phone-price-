# Mobile Price Classification- Final Project

install.packages("caret")
library(caret)
install.packages("psych")
library(psych) # function creates a graph of a correlation matrix, 
# coloring the regions by the level of correlation

# Reading the dataset 
data<-read.csv("train.csv")

# performing EDA 
head(data)

# 1. Dimension of the datasets 
dim(data) # training data has 2000 observations and 21 variables 


# 2. class of the dataset
class(data) # Returns the class of the training data is a dataframe 

# 3. Names of the features/predictors 
names (data)

#4. Summary of the training data 
summary(data)

# Data Cleansing 
# 5. Checking the missing values 
is.na(data) # returns a logical vector if there are NA values in the dataset. 
sum(is.na(data)) # Returns the total number of missing values as 0; Hence there are no missing values in the dataset

data$price_range [1:20] #reading the first 20 values of the response variable 

# Data Analysis
# 6. Finding correlations between the features 
attach(data)
cor(data)
plot(cor(data))

corPlot(data, cex = 1.2)


# High correlation between RAM and price_range 
#3G and 4G
#pc(Primary Camera mega pixels) and fc(Front Camera mega pixels)
#px_weight(Pixel Resolution Width) and px_height(Pixel Resolution Height)
#sc_w(Screen Width of mobile in cm) and sc_h(Screen Height of mobile in cm)


# Finding thd correlation between the above variables 
cor(data$ram, data$price_range) # High correlation as depcited by the plot 
cor(data$pc, data$fc)
cor(data$sc_w, data$sc_h)
cor(data$three_g, data$four_g)

# From the table and heat map, we see that battery_power and px_height is also slightly correlated 
cor(data$battery_power, data$price_range)
cor(data$px_height, data$price_range)


# MACHINE LEARNING: Fitting different models 

# Splitting the data into training and test set 
set.seed(111)
index<- sample(1:nrow(data),size=round(0.7*nrow(data)),replace=F) 
train<- data[index,]
test<-data[-index,]
#1. Fitting SVM on the three correlated variables 

library(e1071)
set.seed(1)
table(train$price_range) # Returns the number of the different price range 

# fitting SVM on the entire dataset

# Changing the response variable to factor before using SVM 
dat<-data.frame(x=train[,c(1:20)], y=as.factor(train$price_range))
svmfit<-svm(y~.,data=dat, kernel="linear", cost=10)
summary(svmfit)
names(svmfit)
ctable<-table(svmfit$fitted,dat$y)
ctable
plot(ctable, col= (factor=c(0,1,2,3)), main=" Confusion Matrix Training")

# Training error 
(8+2+2+4+3)/(2000)
# training error is 0.95% 

#Checking the performance on the test observations

dat.test<-data.frame(x=test[,c(1:20)], y=as.factor(test$price_range))
predict.test<-predict(svmfit, newdata=dat.test)
table2<-table(predict.test, dat.test$y)
table2
(8+4+3+3+1+4)/(149+8+3+137+1+132+4+159)
 # test error is 3.87% 


#  Fitting radial kernel SVM 
svm1 <- svm(y~., data=dat, method="C-classification", kernal="radial", gamma=0.1, cost=10)
summary(svm1) # Gives 1290 support vectors 

table3<-table(svmfit$fitted,dat$y)# Creating the confusion matrix
table3
#training error
(3+1+4+8+4+3)/(3+1+4+8+4+3+149+137+132+159) 

# training error=3.83%
predict.test1<-predict(svm1, newdata=dat.test)
table2<-table(predict.test1, dat.test$y)
table2

#test error
(10+13+23+30+24+8)/(10+13+23+30+24+8+127+110+115+139)


# PCA and Random Forest
library(ggplot2)
names(data)
data$price_range <- factor(data$price_range)
data$three_g <- factor(data$three_g)
data$wifi <- factor(data$wifi)
data$touch_screen <- factor(data$touch_screen)
data$four_g<- factor(data$four_g)
data$dual_sim <- factor(data$dual_sim)
data$blue<- factor(data$blue)

str(data)

#doing stratified sampling tp split the data
#first splitting entire data set based on response 0, 1 , 2, 3 

library(dplyr)
response_count<- data%>%
  count(price_range)
library(ggplot2)
ggplot(response_count,aes(price_range,n,fill=price_range))+geom_col() + ylab("count")+theme_classic()

#as we can see the data is fairly balance and therefore no need to stratify sampling
# simply splitting randomly in training and testing 
set.seed(111)

index<- sample(1:nrow(data),size=round(0.7*nrow(data)),replace=F)
train<- data[index,]
test<-data[-index,]
max(index)
#here we cans see maximum index value is 2000 which aligns with our number of obs
# PCA for EDA 

for (i in 1:ncol(data)){
  data[,i]<- as.numeric(data[,i])
}
str(data)
pca_data<-data[,-21]
pca_1<-prcomp(pca_data,scale=T)
par(mar = c(2, 2, 2, 2))
biplot(pca_1,scale=0)
var<- pca_1$sdev^2
plot(var,xlab ="pc's")
pve=var/sum(var)
cumsum(pve)
plot(cumsum(pve))
plot(pca_1$x[,1],pca_1$x[,2],xlab="PC1")
summary(pca_1)

#corplot
library(corrplot)
corrplot(cor(data),method = "circle")
corela<- data.frame(cor(data))
corela %>%
  select(price_range)%>%
  arrange(desc(price_range))
ggplot(corela,aes(rownames(corela),price_range))+geom_point()+ylim(range(-1:1))
#batery power vs response
ggplot(data,aes(data$price_range,data$battery_power))+geom_col()
# ram and response
ggplot(data,aes(factor(price_range),ram))+geom_col()
str(data)
str(data)
library(randomForest)
set.seed(111)
tuneRF(train[,-21],train[,21],ntreeTry = 400)
random_forest_model<-randomForest(train[,-21],train[,21],mtry = 8,ntree = 400,importance=TRUE)
random_forest_model
random_forest_model$confusion
random_forest_model$err.rate
plot(random_forest_model)
varImpPlot(random_forest_model)
importance(random_forest_model)
importance_var<- data.frame(importance(random_forest_model))
importance_var <- importance_var%>%
  select(MeanDecreaseAccuracy,MeanDecreaseGini)%>%
  arrange(desc(MeanDecreaseGini)) 
importance_var

#important variables
ggplot(importance_var,aes(MeanDecreaseGini,rownames(importance_var),fill=rownames(importance_var)))+geom_col()+theme_classic()+ylab("Features")

# predicting random forest 
yhat.forest<- predict(random_forest_model,newdata=test,type="class")
library(caret)
cf_random_forest<- confusionMatrix(data=yhat.forest,reference = test$price_range)

#final predicted accuracy random forest is 91.1 percent
cf_random_forest$byClass
cf_random_forest$overall
cf_random_forest$table

hh<-data.frame(cf_random_forest$byClass)


#highest accuracy was predicting class 0
plot(cf_random_forest$table,col=factor(c(0,1,2,3)),main="Confusion Matrix plot")
ggplot(hh,aes(rownames(hh),F1,fill=rownames(hh)))+geom_tile()+geom_tile(aes(rownames(hh)))+theme_classic()+ylab("F1 Score")+xlab("Classes")

#ROC
library(mlbench)
library(pROC)
num<- as.numeric(yhat.forest)
roc_multi<-multiclass.roc(test$price_range,num)
auc(roc_multi)

roc_multi$rocs
rs <- roc_multi[['rocs']]
plot.roc(rs[[1]])
sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))

cf_random_forest$overall
cf_random_forest$byClass

# Performing CV for test error 
library(scales)
pie_values<-hh$Prevalence
pie_class<-c(0:3)
pd<-data.frame(cbind(pie_class,round(pie_values,2)))
pd
ggplot(pd,aes(x="",y=V2,fill=factor(pie_class)))+geom_bar(stat="identity",width=1)+coord_polar("y", start =0)+theme_void()+geom_col(color="black")+ geom_text(aes(label = V2), position = position_stack(vjust = 0.5))+ 
  coord_polar(theta="y")

# Plotting Box plots of Important Variables 
ggplot(data,aes(x=factor(price_range),y=ram,fill=price_range))+geom_boxplot()+xlab("Price Range")+theme_classic()
ggplot(data,aes(x=factor(price_range),y=battery_power,fill=price_range))+geom_boxplot()+xlab("Price Range")+theme_classic()
ggplot(data,aes(x=factor(price_range),y=px_height,fill=price_range))+geom_boxplot()+xlab("Price Range")+theme_classic()
ggplot(data,aes(x=factor(price_range),y=px_width,fill=price_range))+geom_boxplot()+xlab("Price Range")+theme_classic()
ggplot(data,aes(x=factor(price_range),y=mobile_wt,fill=price_range))+geom_boxplot()+xlab("Price Range")+theme_classic()


# Fitting Multinomial Logistic Regression 
library(nnet)
set.seed(111)
train$price_range <- relevel(train$price_range,ref="0")
mlogit<-multinom(train$price_range~.,data=train)
summary(mlogit)
#convert from log odds
exp(coef(mlogit))


head(round(fitted(mlogit), 2))
# Predicting the values for train dataset
train.ClassPredicted <- predict(mlogit, newdata = train)

# Building classification table
library(caret)
cf_logit<-confusionMatrix(data=train.ClassPredicted,reference = train$price_range)
cf_logit$table
cf_logit$byClass
cf_logit$overall

# Predicting the class for test dataset
test.ClassPredicted <- predict(mlogit, newdata = test)
cf_logit_test<-confusionMatrix(data=test.ClassPredicted,reference = test$price_range)

# Building classification table
cf_logit_test$byClass
cf_logit_test$overall
cf_logit_test$table

heatmap(cf_logit_test$byClass)

summary(mlogit)$coeff
coef(mlogit)
glm()
z_values <- summary(mlogit)$coefficients / summary(mlogit)$standard.errors
z_values
# p value - 2 tailed z score
p_values<-pnorm(abs(z_values), lower.tail=FALSE)*2
p_values
p_values_df<-data.frame(p_values)
a<-data.frame()

l<-c()
sig_p_values<-p_values_df[,]<0.01
sig_p_values<-data.frame(sig_p_values)
sig_p_values

for (i in 2:ncol(sig_p_values)){
  if (sum(sig_p_values[i])==3){
    l[i]<-colnames(sig_p_values[i])
  }
}
l[!is.na(l)]

1-0.0387


a<- c("Multi Logistic Regression","Random Forest","SVM Kernel Linear")
accuracy<- c(97.5,91.16,96.13)
accuracy<-as.numeric(accuracy)
frame<- cbind(a,accuracy)
frame<- data.frame(frame)
ggplot(frame,aes(x=factor(a),y=accuracy,fill=factor(a)))+geom_col()+xlab("Model") + ylab("Test ccuracy")


































