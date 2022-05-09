# install.packages("readr") # you only need to do this one time on your system
# library(readr)
# mystring <- read_file("json_file")

json_file <-  'data/model-api'
json_data <- readChar(json_file, file.info(json_file)$size)
library("rjson")

out <- lapply(readLines(json_file), fromJSON)
out
#json_file <- "http://api.worldbank.org/country?per_page=10&region=OED&lendingtype=LNX&format=json"
json_list <- fromJSON(paste(readLines(json_file), collapse=""))
json_list2 <- fromJSON(file = json_file)
json_data

df <- do.call("rbind", json_list)
df2 <- do.call("rbindlist", out)
df
json_list
library(jsonlite)
df <- fromJSON(json_data)
#           name  group age (y) height (cm) wieght (kg) score
# 1    Doe, John    Red      24         182        74.8    NA
# 2    Doe, Jane  Green      30         170        70.1   500
# 3  Smith, Joan Yellow      41         169        60.0    NA
# 4   Brown, Sam  Green      22         183        75.0   865
# 5 Jones, Larry  Green      31         178        83.9   221
# 6 Murray, Seth    Red      35         172        76.2   413
# 7    Doe, Jane Yellow      22         164        68.0   902

str(fromJSON(json_file))