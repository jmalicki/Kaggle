
library(arm)
library(caret)
library(gbm)
library(randomForest)
library(caTools)
library(foreach)
library(doMC)
library(e1071)

registerDoMC(cores=4)

#Calc AUC

display_results <- function(train_pred, trainTarget, test_pred, testTarget){
  train_AUC <- colAUC(train_pred, trainTarget)
  test_AUC <- colAUC(test_pred, testTarget)
  cat("\n\n*** what ***\ntraining:")
  print(train_AUC)
  cat("\ntesting:")
  print(test_AUC)
  cat("\n*****************************\n")
  list(train_AUC=train_AUC,test_AUC=test_AUC)
}

getResample <- function(training, pctDeadbeat,pct=1) {
  deadbeat.size <- floor(pct*pctDeadbeat*nrow(training))
  responsible.size <- floor(pct*nrow(training)*(1-pctDeadbeat))

  training.deadbeat <- training[training$SeriousDlqin2yrs == 1,]
  training.responsible <- training[training$SeriousDlqin2yrs == 0,]

  training <- rbind(training.deadbeat[sample(1:nrow(training.deadbeat),
                                             deadbeat.size,
                                             replace=TRUE),],
                    training.responsible[sample(1:nrow(training.responsible),
                                                responsible.size,
                                                replace=TRUE),])
  training
}

buildRFModel <- function(training, pctDeadbeat) {

  # GC multiple times to force memory back to the OS, so we
  # can run multiple processes with as little memory as possible.
  gc(reset=TRUE)
  cat("\n**************\n\nRF pctDeadbeat=",pctDeadbeat,"\n\n***********\n\n")
  RF <- foreach(ntree=rep(200,16), .combine=combine,
                .multicombine=TRUE,
                .packages="randomForest") %dopar% {
                  training.SeriousDlqin2yrs <- training$SeriousDlqin2yrs
                  training <- training[,-c(1,2)]
                  classwt <- c((1-pctDeadbeat)/sum(training.SeriousDlqin2yrs == 0),
                               pctDeadbeat/sum(training.SeriousDlqin2yrs == 1)) *
                                 nrow(training)
                  
                  randomForest(training,
                               factor(training.SeriousDlqin2yrs),
                               ntree=ntree,
                               strata=factor(training.SeriousDlqin2yrs),
                               do.trace=TRUE, importance=TRUE, forest=TRUE,
                               replace=TRUE, classwt=classwt)
                }
  RF
}

buildRFModelEnsemble <- function(training) {
  # rf2 was less important in final model
  rfensemble<-lapply(list(rf1=0.1,
                          rf2=0.25,
                          rf3=0.375,
                          rf4=0.5,
                          rf5=0.625),
                     function(pctDeadbeat) buildRFModel(training, pctDeadbeat))
  save(rfensemble,file='rfensemble.rda')
  rfensemble
}

buildSVM <- function(training, cost=1) {
  response <- factor(training$SeriousDlqin2yrs)
  weight <- sum(training$SeriousDlqin2yrs)/nrow(training)
  training <- training[,-c(1,2)]
  gc()
  weight <- c(1/(1-weight), 1/weight)
  names(weight) <- levels(response)

  # already scaled
  svm(training, response, scale=FALSE, type='C-classification',
      kernel='radial', cachesize=4000, probability=TRUE,
      class.weights=weight, cost=cost)
}

buildSVMEnsemble <- function(training) {
  gc(reset=TRUE)
  mclapply(list(svm1=0.05, svm3=0.1, svm4=0.5, svm5=1),
           function(cost) buildSVM(training, cost=cost))
}

buildGBMModel <- function(training, pctDeadbeat) {
  training$X <- NULL
  weight <- sum(training$SeriousDlqin2yrs)/nrow(training)
  weights <- c((1-pctDeadbeat)/(1-weight),pctDeadbeat/weight)[1+training$SeriousDlqin2yrs]
  GB <- gbm(training$SeriousDlqin2yrs ~ ., data=training, n.trees=1000,
            keep.data=FALSE, shrinkage=0.01, bag.fraction=0.3,
            weights = weights,
            interaction.depth=10)
  GB
}

buildGBMModelEnsemble <- function(training) {
  gc(reset=TRUE)
  # gb1, gb5 commented out out because they were less important
  # in final model
  mclapply(list(gb1=0.1,
                gb2=0.15,
                gb3=0.25,
                gb4=0.375,
                gb5=0.5,
                gb6=0.625),
           function(pctDeadbeat)
           buildGBMModel(training, pctDeadbeat))
}

buildLinModel <- function(training) {
  bayesglm(SeriousDlqin2yrs ~ .,
           data = training[,-1], prior.scale=0.7,
           family=binomial, scaled=FALSE,
           n.iter=200)
}

buildLinEnsemble <- function(GBs,RFs,lin,training,preprocess) {
  # fixme: parallelize below
  submodels<-list(rfs=RFs,gbs=GBs,
                  #svms=SVMs,
                  lin=lin,
                  preprocess=preprocess)

  z <- predictSubModels(submodels, training)
  z$SeriousDlqin2yrs <- training$SeriousDlqin2yrs

  cat("Fitting final ensemble\n")
  submodels$ensemble<-bayesglm(SeriousDlqin2yrs ~ ., family=binomial, data=z,
                        prior.df=Inf)
  class(submodels) <- 'GiveMeCredit'
  submodels
}

buildSubModels <- function(training, testing) {
  cat("Centering and scaling\n")
  preprocess <- preProcess(rbind(testing,training))

  training.SeriousDlqin2yrs <- training$SeriousDlqin2yrs
  training <- predict(preprocess, training)
  print(summary(training$SeriousDlqin2yrs <- training.SeriousDlqin2yrs))
  training.SeriousDlqin2yrs <- NULL
  
  testing.SeriousDlqin2yrs <- testing$SeriousDlqin2yrs
  testing <- predict(preprocess, testing)
  testing$SeriousDlqin2yrs <- testing.SeriousDlqin2yrs
  testing.SeriousDlqin2yrs <- NULL
  
  cat("Building models\n")
  gc(reset=TRUE)

  # mclapply uses serialize, which I think can't deal with >2gb data,
  # so we just can't use mclapply or papply here :(
  submodels <- lapply(list(rfs=buildRFModelEnsemble,
                           gbs=buildGBMModelEnsemble,
                           #svms=buildSVMEnsemble,
                           lin=buildLinModel),
                      function(f) f(training))
  submodels$preprocess <- preprocess
  submodels
}

buildModels <- function(submodels, testing) {
  # We train ensemble on testing very purposefully,
  # because random forest fits training data too well by its very nature,
  # making an ensemble there useless.
  cat("Building ensemble\n")
  testing.SeriousDlqin2yrs <- testing$SeriousDlqin2yrs
  testing <- predict(submodels$preprocess, testing)
  testing$SeriousDlqin2yrs <- testing.SeriousDlqin2yrs

  r <- buildLinEnsemble(submodels$gbs, submodels$rfs, #submodels$svms,
                        submodels$lin, testing,
                        submodels$preprocess)
  print(r)

  r
}

print.GiveMeCredit <- function(m) print(m$ensemble)
  
predictSubModels <- function(m, d) {
  d.SeriousDlqin2yrs <- d$SeriousDlqin2yrs
  d <- predict(m$preprocess, d)
  d$SeriousDlqin2yrs <- d.SeriousDlqin2yrs

  gc()
  gbs <- lapply(m$gbs, function(subm) 1/(1+exp(-predict(subm, d[,-c(1,2)], n.tree=1000))))
  gc()
  rfs <- lapply(m$rfs, function(subm) predict(subm, d[,-c(1,2)], type='prob')[,2])
  gc()
  #svms <- mclapply(m$svms, function(subm) attr(predict(subm, testing[1:100,-c(1,2)], probability=TRUE),"probabilities")[,1])

  cbind(data.frame(gbs),
        data.frame(rfs),
        #data.frame(svms),
        lin=1/(1+exp(-predict(m$lin, d[,-c(1,2)]))))
}

predict.GiveMeCredit <- function(m, d) {
  z <- predictSubModels(m, d)
  z$ensemble <- predict(m$ensemble, z, type='response')
  z
}

evalResults.GiveMeCredit <- function(m, training, test) {  
  trainTarget <- training$SeriousDlqin2yrs
  train_pred <-predict(m, training)
  testTarget <- test$SeriousDlqin2yrs
  test_pred <- predict(m, test)
  display_results(train_pred, trainTarget, test_pred, testTarget)
}

xform_data <- function(x) {
  x$UnknownNumberOfDependents <- as.integer(is.na(x$NumberOfDependents))
  x$UnknownMonthlyIncome <- as.integer(is.na(x$MonthlyIncome))

  x$NoDependents <- as.integer(x$NumberOfDependents == 0)
  x$NoDependents[is.na(x$NoDependents)] <- 0

  x$NumberOfDependents[x$UnknownNumberOfDependents==1] <- 0

  x$NoIncome <- as.integer(x$MonthlyIncome == 0)
  x$NoIncome[is.na(x$NoIncome)] <- 0

  x$MonthlyIncome[x$UnknownMonthlyIncome==1] <- 0

  x$ZeroDebtRatio <- as.integer(x$DebtRatio == 0)
  x$UnknownIncomeDebtRatio <- x$DebtRatio
  x$UnknownIncomeDebtRatio[x$UnknownMonthlyIncome == 0] <- 0
  x$DebtRatio[x$UnknownMonthlyIncome == 1] <- 0

  x$WeirdRevolvingUtilization <- x$RevolvingUtilizationOfUnsecuredLines
  x$WeirdRevolvingUtilization[!(log(x$RevolvingUtilizationOfUnsecuredLines) > 3)] <- 0
  x$ZeroRevolvingUtilization <- as.integer(x$RevolvingUtilizationOfUnsecuredLines == 0)
  x$RevolvingUtilizationOfUnsecuredLines[log(x$RevolvingUtilizationOfUnsecuredLines) > 3] <- 0
  
  x$Log.Debt <- log(pmax(x$MonthlyIncome, rep(1, nrow(x))) * x$DebtRatio)
  x$Log.Debt[!is.finite(x$Log.Debt)] <- 0
  
  x$RevolvingLines <- x$NumberOfOpenCreditLinesAndLoans - x$NumberRealEstateLoansOrLines

  x$HasRevolvingLines <- as.integer(x$RevolvingLines > 0)
  x$HasRealEstateLoans <- as.integer(x$NumberRealEstateLoansOrLines > 0)
  x$HasMultipleRealEstateLoans <- as.integer(x$NumberRealEstateLoansOrLines > 2)
  x$EligibleSS <- as.integer(x$age >= 60)
  x$DTIOver33 <- as.integer(x$NoIncome == 0 & x$DebtRatio > 0.33)
  x$DTIOver43 <- as.integer(x$NoIncome == 0 & x$DebtRatio > 0.43)
  x$DisposableIncome <- (1 - x$DebtRatio) * x$MonthlyIncome
  x$DisposableIncome[x$NoIncome == 1] <- 0
  
  x$RevolvingToRealEstate <- x$RevolvingLines / (1 + x$NumberRealEstateLoansOrLines)

  x$NumberOfTime30.59DaysPastDueNotWorseLarge <- as.integer(x$NumberOfTime30.59DaysPastDueNotWorse > 90)
  x$NumberOfTime30.59DaysPastDueNotWorse96 <- as.integer(x$NumberOfTime30.59DaysPastDueNotWorse == 96)
  x$NumberOfTime30.59DaysPastDueNotWorse98 <- as.integer(x$NumberOfTime30.59DaysPastDueNotWorse == 98)
  x$Never30.59DaysPastDueNotWorse <- as.integer(x$NumberOfTime30.59DaysPastDueNotWorse == 0)
  x$NumberOfTime30.59DaysPastDueNotWorse[x$NumberOfTime30.59DaysPastDueNotWorse > 90] <- 0

  x$NumberOfTime60.89DaysPastDueNotWorseLarge <- as.integer(x$NumberOfTime60.89DaysPastDueNotWorse > 90)
  x$NumberOfTime60.89DaysPastDueNotWorse96 <- as.integer(x$NumberOfTime60.89DaysPastDueNotWorse == 96)
  x$NumberOfTime60.89DaysPastDueNotWorse98 <- as.integer(x$NumberOfTime60.89DaysPastDueNotWorse == 98)
  x$Never60.89DaysPastDueNotWorse <- as.integer(x$NumberOfTime60.89DaysPastDueNotWorse == 0)
  x$NumberOfTime60.89DaysPastDueNotWorse[x$NumberOfTime60.89DaysPastDueNotWorse > 90] <- 0
  
  x$NumberOfTimes90DaysLateLarge <- as.integer(x$NumberOfTimes90DaysLate > 90)
  x$NumberOfTimes90DaysLate96 <- as.integer(x$NumberOfTimes90DaysLate == 96)
  x$NumberOfTimes90DaysLate98 <- as.integer(x$NumberOfTimes90DaysLate == 98)
  x$Never90DaysLate <- as.integer(x$NumberOfTimes90DaysLate == 0)
  x$NumberOfTimes90DaysLate[x$NumberOfTimes90DaysLate > 90] <- 0

  x$IncomeDivBy10 <- as.integer(x$MonthlyIncome %% 10 == 0)
  x$IncomeDivBy100 <- as.integer(x$MonthlyIncome %% 100 == 0)
  x$IncomeDivBy1000 <- as.integer(x$MonthlyIncome %% 1000 == 0)
  x$IncomeDivBy5000 <- as.integer(x$MonthlyIncome %% 5000 == 0)
  x$Weird0999Utilization <- as.integer(x$RevolvingUtilizationOfUnsecuredLines == 0.9999999)
  x$FullUtilization <- as.integer(x$RevolvingUtilizationOfUnsecuredLines == 1)
  x$ExcessUtilization <- as.integer(x$RevolvingUtilizationOfUnsecuredLines > 1)

  x$NumberOfTime30.89DaysPastDueNotWorse <- x$NumberOfTime30.59DaysPastDueNotWorse + x$NumberOfTime60.89DaysPastDueNotWorse
  x$Never30.89DaysPastDueNotWorse <- x$Never60.89DaysPastDueNotWorse * x$Never30.59DaysPastDueNotWorse
  
  x$NumberOfTimesPastDue <- x$NumberOfTime30.59DaysPastDueNotWorse + x$NumberOfTime60.89DaysPastDueNotWorse + x$NumberOfTimes90DaysLate
  x$NeverPastDue <- x$Never90DaysLate * x$Never60.89DaysPastDueNotWorse * x$Never30.59DaysPastDueNotWorse
  x$Log.RevolvingUtilizationTimesLines <- log1p(x$RevolvingLines * x$RevolvingUtilizationOfUnsecuredLines)

  x$Log.RevolvingUtilizationOfUnsecuredLines <- log(x$RevolvingUtilizationOfUnsecuredLines)
  x$Log.RevolvingUtilizationOfUnsecuredLines[is.na(x$Log.RevolvingUtilizationOfUnsecuredLines)] <- 0
  x$Log.RevolvingUtilizationOfUnsecuredLines[!is.finite(x$Log.RevolvingUtilizationOfUnsecuredLines)] <- 0
  x$RevolvingUtilizationOfUnsecuredLines <- NULL
  
  x$DelinquenciesPerLine <- x$NumberOfTimesPastDue / x$NumberOfOpenCreditLinesAndLoans
  x$DelinquenciesPerLine[x$NumberOfOpenCreditLinesAndLoans == 0] <- 0
  x$MajorDelinquenciesPerLine <- x$NumberOfTimes90DaysLate / x$NumberOfOpenCreditLinesAndLoans
  x$MajorDelinquenciesPerLine[x$NumberOfOpenCreditLinesAndLoans == 0] <- 0
  x$MinorDelinquenciesPerLine <- x$NumberOfTime30.89DaysPastDueNotWorse / x$NumberOfOpenCreditLinesAndLoans
  x$MinorDelinquenciesPerLine[x$NumberOfOpenCreditLinesAndLoans == 0] <- 0

  # Now delinquencies per revolving
  x$DelinquenciesPerRevolvingLine <- x$NumberOfTimesPastDue / x$RevolvingLines
  x$DelinquenciesPerRevolvingLine[x$RevolvingLines == 0] <- 0
  x$MajorDelinquenciesPerRevolvingLine <- x$NumberOfTimes90DaysLate / x$RevolvingLines
  x$MajorDelinquenciesPerRevolvingLine[x$RevolvingLines == 0] <- 0
  x$MinorDelinquenciesPerRevolvingLine <- x$NumberOfTime30.89DaysPastDueNotWorse / x$RevolvingLines
  x$MinorDelinquenciesPerRevolvingLine[x$RevolvingLines == 0] <- 0

  
  x$Log.DebtPerLine <- x$Log.Debt - log1p(x$NumberOfOpenCreditLinesAndLoans)
  x$Log.DebtPerRealEstateLine <- x$Log.Debt - log1p(x$NumberRealEstateLoansOrLines)
  x$Log.DebtPerPerson <- x$Log.Debt - log1p(x$NumberOfDependents)
  x$RevolvingLinesPerPerson <- x$RevolvingLines / (1 + x$NumberOfDependents)
  x$RealEstateLoansPerPerson <- x$NumberRealEstateLoansOrLines / (1 + x$NumberOfDependents)
  x$UnknownNumberOfDependents <- as.integer(x$UnknownNumberOfDependents)
  x$YearsOfAgePerDependent <- x$age / (1 + x$NumberOfDependents)

  x$Log.MonthlyIncome <- log(x$MonthlyIncome)
  x$Log.MonthlyIncome[!is.finite(x$Log.MonthlyIncome)|is.na(x$Log.MonthlyIncome)] <- 0
  x$MonthlyIncome <- NULL
  x$Log.IncomePerPerson <- x$Log.MonthlyIncome - log1p(x$NumberOfDependents)
  x$Log.IncomeAge <- x$Log.MonthlyIncome - log1p(x$age)
  
  x$Log.NumberOfTimesPastDue <- log(x$NumberOfTimesPastDue)
  x$Log.NumberOfTimesPastDue[!is.finite(x$Log.NumberOfTimesPastDue)] <- 0
  
  x$Log.NumberOfTimes90DaysLate <- log(x$NumberOfTimes90DaysLate)
  x$Log.NumberOfTimes90DaysLate[!is.finite(x$Log.NumberOfTimes90DaysLate)] <- 0

  x$Log.NumberOfTime30.59DaysPastDueNotWorse <- log(x$NumberOfTime30.59DaysPastDueNotWorse)
  x$Log.NumberOfTime30.59DaysPastDueNotWorse[!is.finite(x$Log.NumberOfTime30.59DaysPastDueNotWorse)] <- 0
  
  x$Log.NumberOfTime60.89DaysPastDueNotWorse <- log(x$NumberOfTime60.89DaysPastDueNotWorse)
  x$Log.NumberOfTime60.89DaysPastDueNotWorse[!is.finite(x$Log.NumberOfTime60.89DaysPastDueNotWorse)] <- 0

  x$Log.Ratio90to30.59DaysLate <- x$Log.NumberOfTimes90DaysLate - x$Log.NumberOfTime30.59DaysPastDueNotWorse
  x$Log.Ratio90to60.89DaysLate <- x$Log.NumberOfTimes90DaysLate - x$Log.NumberOfTime60.89DaysPastDueNotWorse

  x$AnyOpenCreditLinesOrLoans <- as.integer(x$NumberOfOpenCreditLinesAndLoans > 0)
  x$Log.NumberOfOpenCreditLinesAndLoans <- log(x$NumberOfOpenCreditLinesAndLoans)
  x$Log.NumberOfOpenCreditLinesAndLoans[!is.finite(x$Log.NumberOfOpenCreditLinesAndLoans)] <- 0
  x$Log.NumberOfOpenCreditLinesAndLoansPerPerson <- x$Log.NumberOfOpenCreditLinesAndLoans - log1p(x$NumberOfDependents)

  x$Has.Dependents <- as.integer(x$NumberOfDependents > 0)
  x$Log.HouseholdSize <- log1p(x$NumberOfDependents)
  x$NumberOfDependents <- NULL

  x$Log.DebtRatio <- log(x$DebtRatio)
  x$Log.DebtRatio[!is.finite(x$Log.DebtRatio)] <- 0
  x$DebtRatio <- NULL

  x$Log.DebtPerDelinquency <- x$Log.Debt - log1p(x$NumberOfTimesPastDue)
  x$Log.DebtPer90DaysLate <- x$Log.Debt - log1p(x$NumberOfTimes90DaysLate)

  
  x$Log.UnknownIncomeDebtRatio <- log(x$UnknownIncomeDebtRatio)
  x$Log.UnknownIncomeDebtRatio[!is.finite(x$Log.UnknownIncomeDebtRatio)] <- 0
  x$IntegralDebtRatio <- NULL
  x$Log.UnknownIncomeDebtRatioPerPerson <- x$Log.UnknownIncomeDebtRatio - x$Log.HouseholdSize
  x$Log.UnknownIncomeDebtRatioPerLine <- x$Log.UnknownIncomeDebtRatio - log1p(x$NumberOfOpenCreditLinesAndLoans)
  x$Log.UnknownIncomeDebtRatioPerRealEstateLine <- x$Log.UnknownIncomeDebtRatio - log1p(x$NumberRealEstateLoansOrLines)
  x$Log.UnknownIncomeDebtRatioPerDelinquency <- x$Log.UnknownIncomeDebtRatio - log1p(x$NumberOfTimesPastDue)
  x$Log.UnknownIncomeDebtRatioPer90DaysLate <- x$Log.UnknownIncomeDebtRatio - log1p(x$NumberOfTimes90DaysLate)

  x$Log.NumberRealEstateLoansOrLines <- log(x$NumberRealEstateLoansOrLines)
  x$Log.NumberRealEstateLoansOrLines[!is.finite(x$Log.NumberRealEstateLoansOrLines)] <- 0
  x$NumberRealEstateLoansOrLines <- NULL
  
  x$NumberOfOpenCreditLinesAndLoans <- NULL
  
  x$NumberOfTimesPastDue <- NULL
  x$NumberOfTimes90DaysLate <- NULL
  x$NumberOfTime30.59DaysPastDueNotWorse <- NULL
  x$NumberOfTime60.89DaysPastDueNotWorse <- NULL

  x$LowAge <- as.integer(x$age < 18)
  x$Log.age <- log(x$age - 17)
  x$Log.age[x$LowAge == 1] <- 0
  x$age <- NULL
  
  x
}

getData <- function() {
  training1 <- read.csv("cs-training.csv")
  training <- training1[c(1:125000),]
  test <- training1[c(125001:150000),]
  list(training=training,testing=test)
}

makeSubmission <- function(RF) {
  test <- xform_data(read.csv("cs-test.csv"))
  gc()
  pred <- data.frame(predict(RF,test[,-c(1,2)], type='prob')[,2])
  gc()
  
  write.csv(pred, file="PR001.csv")
}

makeSubmission.GiveMeCredit <- function(m) {
  test <- xform_data(read.csv("cs-test.csv"))
  pred<-predict(m, test)
  pred<-data.frame(Row_ID=test[,1],ensemble=pred$ensemble)
  pred$ensemble<-1/(1+exp(-pred$ensemble))
  write.csv(pred, file="PRlin.csv", row.names=FALSE)
}


makeSplom <- function(x) {
  x$MonthlyIncome <- log(1+x$MonthlyIncome)
  x$DebtToIncome <- log(1+x$DebtRatio)
  splom(x)
}

doItAll <- function() {
  cat("Getting data\n")
  print(system.time(d <- getData()))
  cat("Transforming data\n")
  print(system.time({
    d$training <- xform_data(d$training)
    d$testing <- xform_data(d$testing)
  }))
  cat("Building Model\n")
  print(system.time(
                    subm <- buildSubModels(d$training, d$testing)
        ))
  m <- buildModels(subm, d$testing)

  model.results <- list(d=d, m=m)
  subm <- NULL
  m <- NULL
  d <- NULL
  gc()
  print(system.time(save(model.results, file='submission.rda')))
  gc()
  
  evalResults.GiveMeCredit(model.results$m, model.results$d$training,
                           model.results$d$testing)
  gc()
  
  cat("Making Submission\n")
  print(system.time(makeSubmission.GiveMeCredit(model.results$m)))
  cat("Saving\n")
  gc()
  model.results
}
