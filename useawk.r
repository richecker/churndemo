    # specify output file and width
#postscript(file="useawk.ps", width=40)
#pdf        (file="useawk.pdf",width=40)
print('reading in data...')
    # read in two files
mtx1 <- read.table("data/useawk-d1.data")
mtx2 <- read.table("data/useawk-d2.data")

    #init the arrays to hold the mean values
    #we will need 200 slots at most
meanArr1<-rep(NA, 200)
meanArr2<-rep(NA, 200)

    # the x label values
x<-rep(NA, 200)

    # leave the first spot for 0
x[1] <- 0

for(i in 1:200){ 
    meanArr1[i] <- mean(mtx1[((i-1)*10):(i*10),5])
        # i+1 because we left the first slot for 0
    x[i+1] = mtx1[i,1]
}
for(i in 1:200){
  meanArr2[i] <- mean(mtx2[((i-1)*10):(i*10),5]) 
}

plot(meanArr1,col="gray", #line colour
        xlim=c(0,160), 
        ylim=c(0,10000), 
        t="l",      # we want a line graph
        lwd=1.5,    # how thick the line should be
        ylab="Number of new terms",
        xlab="Number of documents",
        xaxt="n", yaxt="n")

    # plt the x axis array using x
axis(1, 1:201, x)
    # plt the y axis 
axis(2)

    # make sure the two graphs appear on the same axis
par(new=TRUE) 

plot(meanArr2,col="black", 
        xlim=c(0,160), 
        ylim=c(0,10000), t="l",
        xlab="", # we left these empty because the previous graph has
        ylab="", # done them
        lwd=1.5,
        xaxt="n", 
        yaxt="n")

axis(1, at=NULL, labels=NA)
axis(2)

    # draw the legend
legend("topright", c("Without stemming","With stemming"), 
        col=c("black", "gray"), lty=c(1,1), lwd="2")
print('plots in Rplots.pdf file')

#dev.off()  # This is only needed if you use pdf/postscript in interactive mode
