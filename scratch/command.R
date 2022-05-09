list.of.packages <- c("reticulate")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
library(reticulate)

system("echo $PATH")
system("echo $LD_LIBRARY_PATH")

system("export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH")
system("echo $LD_LIBRARY_PATH")

# py_run_file("code/BaselineROC.py")
