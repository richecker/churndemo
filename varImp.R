library(randomForest)
library(caret)
library(ggplot2)

args <- commandArgs(trailingOnly=TRUE)
p1 <- args[1]
p2 <- args[2]

formula <- paste0(p2, " ~ .")

df <- read.csv(file=p1, header=TRUE, sep=",")
output.forest <- randomForest(as.formula(formula), data = df, na.action=na.pass)
imp <- varImp(output.forest)
imp$cols <- rownames(imp)

ggplot(data=imp, aes(x=reorder(cols, -Overall), y=Overall)) +
  geom_bar(stat="identity", color = 'blue', fill= 'blue', width=0.5) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Variable Importance Plot") +
  xlab("Inputs") + 
  ylab("Importance \n (Mean Decrease Gini)")
  
#ggsave("/mnt/output/VarImpPlot.png")

imp.top <- imp[order(imp$Overall, decreasing = TRUE),][1:1,]
diagnostics = list("Top Predictor" = imp.top$cols)
library(jsonlite)
fileConn<-file("dominostats.json")
writeLines(toJSON(diagnostics), fileConn)
close(fileConn)
