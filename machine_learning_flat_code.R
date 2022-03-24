## Initialization ---------------------------------------------------------
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

#  Color spectrum
pal <- c('#b2df8a','#33a02c','#ff7f00','#cab2d6','#b15928',
         '#fdbf6f','#a6cee3','#fb9a99','#1f78b4','#e31a1c')

set.seed(3)
pal <- sample(pal,length(mdls),F)

# Modelling ---------------------------------------------------------------

rawdata <- iris

# Model inputs
yvar <- "Sepal.Length"
xvars <- c("Sepal.Width","Petal.Length","Petal.Width")
testsize <- 20

chk_logY <- FALSE # Is log applied to Y

dataTrain <- NULL
dataTest <- NULL

modelType <- 'Regression'
mdls <- reg.mdls

# extract y and X from raw data
y <- rawdata[,yvar]
X <-  rawdata[,xvars]

# deal with NA values
yi <- !is.na(y)
Xi <- complete.cases(X)
df2 <- cbind(y,X)[yi&Xi,]
c <- class(df2$y)

lvls <- length(unique(df2$y))

if(lvls < 10|(c!='numeric'&c!='integer')){
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
  'glm'=list(form=y ~ .,
             data = dataTrain,
             preProcess = c('scale','center'),
             method = 'glm',
             trControl = fitControl,
             tuneGrid=NULL),
  'nb'=list(form=y ~ .,
            data = dataTrain,
            preProcess = c('scale','center'),
            method = 'nb',
            trControl = fitControl,
            tuneGrid=tuneParams[['nb']]),
  'gam'=list(form=y ~ .,
             data = dataTrain,
             preProcess = c('scale','center'),
             method = 'gam',
             trControl = fitControl)
)

tune <- lapply(mdls, function(m){
  do.call('train', trainArgs[[m]])
})

names(tune) <- mdls
CVtune <- tune

fits <- CVtune

getRes <- function(i){
  name <- names(fits)[i]
  res <- fits[[i]]$results
  df <- res[,-1]
  #model <- paste(name, res$C, round(res$RMSE,5), sep = "-")
  model <- apply(res,1,function(r) paste(r[1:(ncol(res)-4)],collapse = '-')) %>% 
    paste(name,.,sep='-')
  cbind.data.frame(model,df,name=name[[1]],stringsAsFactors =F)
}

df <- plyr::ldply(1:length(fits),getRes)

if(modelType=='Regression'){
  df$rank <- rank(rank(df$RMSE)+rank(1-df$Rsquared),ties.method = 'first')
} else {
  df$rank <- rank(rank(1-df$Accuracy)+rank(1-df$Kappa),ties.method = 'first')
}
df[2:5] <- round(df[2:5],3)

CVres <- df[order(df$rank),]

# Plots for regression models ---------------------------------------------

resdf <- CVres
type <- modelType

resdf$model <- factor(resdf$model, levels = rev(resdf$model[resdf$rank]))

ggplot(resdf, aes(x=model,color=name))+
  geom_errorbar(aes(ymin = RMSE - RMSESD, ymax = RMSE + RMSESD), size=1)+
  geom_point(aes(y=RMSE),size=3)+
  scale_color_manual(values=pal)+
  coord_flip()+
  theme_bw()+
  xlab('')+
  theme(legend.position='none') -> p1

ggplot(resdf,aes(x=model,color=name))+
  geom_errorbar(aes(ymin = Rsquared - RsquaredSD, ymax = Rsquared + RsquaredSD), size=1)+
  geom_point(aes(y=Rsquared),size=3)+
  scale_color_manual(values=pal)+
  coord_flip()+
  theme_bw()+
  xlab('')+
  theme(legend.position='none') -> p2

gridExtra::grid.arrange(p2,p1,ncol=2)

