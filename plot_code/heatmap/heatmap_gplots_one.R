library("gplots")
library(ggplot2)
setwd("D://")
png(file="S6_3.png")### ±£´æÍ¼Æ¬
color.palette  <- colorRampPalette(c("#eff3ff","#bdd7e7","#6baed6","#2171b5"))
opar<-par(no.readonly=T)
par(mfrow=c(2,1));
test<-read.csv("C:/Users/Lenovo/Desktop/XXXplot/S6_predict_prob.csv",header =T,sep=",",check.names = F)
chart1 <- heatmap.2(
round(cor(test),2), trace="none",scale = "none",col=color.palette,cexRow=2,cexCol=2,
offsetRow=-0.2, offsetCol=-0.2,srtCol=315,adjCol=c(0,1),density.info = "none",main=" ",margins=c(10,10),key.title="A")#keysize = 1,main="G.subterraneus"

dev.off()
