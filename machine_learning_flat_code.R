## Initialization ---------------------------------------------------------

library(shiny)
require(shinyBS)
require(shinydashboard)
require(shinyjs)
require(caret)
require(plyr)
require(dplyr)
require(tidyr)
require(Cairo)
require(raster)
require(gstat)
require(wesanderson)
require(nnet)
require(randomForest)
require(ggplot2)

# car, foreach, methods, plyr, nlme, reshape2, stats, stats4, utils, grDevices

# Not all of these are required but shinyapps.io was crashing and 
# importing one of these solved the issue
require(kernlab)
require(klaR)
require(vcd)
require(e1071)
require(gam)
require(ipred)
require(MASS)
require(ellipse)
require(mda)
require(mgcv)
require(mlbench)
require(party)
require(MLmetrics)
require(Cubist)
require(testthat)

data(meuse)

dmnds <- diamonds#[sample(1:nrow(diamonds),1e3),]

# leaf <- read.csv('/Users/davesteps/Desktop/kaggle_data/leaf/train.csv')

datasets <- list(
  'iris'=iris,
  'cars'=mtcars,
  'meuse'=meuse,
  'diamonds'=data.frame(dmnds),
  'Boston'=Boston
  # 'leaf'=leaf
  # 'midwest'=data.frame(midwest),
  # 'mpg'=data.frame(mpg),
  # 'msleep'=data.frame(msleep),
  # 'txhousing'=data.frame(txhousing)
)

tuneParams <- list(
  'svmLinear'=data.frame(C=c(0.01,0.1,1)),
  'svmPoly'= expand.grid(degree=1:3,scale=c(0.01,0.1),C=c(0.25,0.5,1)),
  'nnet'=expand.grid(size=c(1,3,5),decay=c(0.01,0.1,1)),
  'rf'=data.frame(mtry=c(2,3,4)),
  'knn'=data.frame(k=c(1,3,5,7,9)),
  'nb'=expand.grid(usekernel=c(T,F),adjust=c(0.01,0.1,1),fL=c(0.01,0.1,1)),
  'glm'=NULL#data.frame()
)


mdls <- list('svmLinear'='svmLinear',
             'svmPoly'='svmPoly',
             'Neural Network'='nnet',
             'randomForest'='rf',
             'k-NN'='knn',
             'Naive Bayes'='nb',
             'GLM'='glm',
             'GAM'='gam')

# Multinomial 
mdli <- list(
  'Regression'=c(T,T,T,T,T,F,T,F),
  'Classification'=c(T,T,T,T,T,T,F,F)
)  

reg.mdls <- mdls[mdli[['Regression']]]
cls.mdls <- mdls[mdli[['Classification']]]

# Modelling -------------------------------------------------------------------

CVtune <- readRDS('initState.Rdata')
makeReactiveBinding('CVtune')

rawdata <- iris

# Model inputs
yvar <- "Species"
xvars <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")
testsize <- 30

dataTrain <- NULL
dataTest <- NULL

makeReactiveBinding('dataTrain')
makeReactiveBinding('dataTest')
modelType <- 'Regression'
makeReactiveBinding('modelType')

# extract y and X from raw data
y <- isolate(rawdata[,yvar])
X <-  isolate(rawdata[,xvars])

# deal with NA values
yi <- !is.na(y)
Xi <- complete.cases(X)

df2 <- cbind(y,X)[yi&Xi,]

c <- class(df2$y)

lvls <- length(unique(df2$y))

if(lvls<10|(c!='numeric'&c!='integer')){
  modelType <-'Classification'
  df2$y <- factor(df2$y)
} else {
  modelType <-'Regression'
  if(chk_logY==TRUE){df2$y <- log(df2$y+0.1)}
}

trainIndex <- createDataPartition(df2$y,
                                  p = 1-(testsize/100),
                                  list = FALSE,
                                  times = 1)

dataTrain <- df2[ trainIndex,]
dataTest  <- df2[-trainIndex,]

fitControl <- trainControl(method = "cv", savePredictions = T, number = 3)

trainArgs <- list(
  'svmLinear'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'svmLinear',
                   trControl = fitControl,
                   tuneGrid=tuneParams[['svmLinear']]),
  'svmPoly'= list(form=y ~ .,
                  data = dataTrain,
                  preProcess = c('scale','center'),
                  method = 'svmPoly',
                  trControl = fitControl,
                  tuneGrid=tuneParams[['svmPoly']]),
  'nnet'=list(form=y ~ .,
              data = dataTrain,
              preProcess = c('scale','center'),
              method = 'nnet',
              trControl = fitControl,
              tuneGrid=tuneParams[['nnet']],
              linout=T),
  'rf'=list(form=y ~ .,
            data = dataTrain,
            preProcess = c('scale','center'),
            method = 'rf',
            trControl = fitControl,
            tuneGrid=tuneParams[['rf']],
            ntree=1e3),
  'knn'=list(form=y ~ .,
             data = dataTrain,
             preProcess = c('scale','center'),
             method = 'knn',
             trControl = fitControl,
             tuneGrid=tuneParams[['knn']]),
  'nb'=list(form=y ~ .,
            data = dataTrain,
            preProcess = c('scale','center'),
            method = 'nb',
            trControl = fitControl,
            tuneGrid=tuneParams[['nb']]),
  'glm'=list(form=y ~ .,
             data = dataTrain,
             preProcess = c('scale','center'),
             method = 'glm',
             trControl = fitControl,
             tuneGrid=NULL),
  'gam'=list(form=y ~ .,
             data = dataTrain,
             preProcess = c('scale','center'),
             method = 'gam',
             trControl = fitControl)
)

tune <- lapply(mdls,function(m){
  do.call('train',trainArgs[[m]])
})



