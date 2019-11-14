###L1_L2
library(ggplot2)
library(grid)
###图1
dat <- read.csv("C:/Users/Lenovo/Desktop/Bioinformatics-master/plot/scatter/L1_L2.csv")
data1 <- data.frame(dat)
###> coef(lm(Negative ~ Positive, data = dat))###计算最佳拟合直线的斜率和直径
###  (Intercept)    Positive 
###  0.1935481   0.4202044 
data1.lm <- lm(Negative ~ Positive, data = data1)
p1 <- ggplot(data1,aes(x = Positive,y = Negative,colour=L_length,shape=value))+
geom_point(alpha = 0.7,size=3)+
facet_wrap(~L_length)+
guides(colour = "none")+
geom_abline(intercept = coef(data1.lm)[1],slope=coef(data1.lm)[2],linetype="longdash")+
labs(tag = "A",x="The value of Positive Samples",y="The value of Negative Samples")+
theme(
axis.text = element_text(colour="black"), ###设置坐标轴文本的大小
axis.title = element_text(size = rel(1.2),colour="black"), ###设置坐标轴标签的大小
panel.background = element_rect(fill = "white", colour = "black"),###将面板背景变空白
panel.grid.major = element_blank(),  
panel.grid.minor = element_blank(),
axis.title.x = element_text(vjust=-1), ###x轴标签距离x轴的大小
axis.title.y = element_text(vjust=4), ###y轴标签距离y轴的大小
axis.ticks = element_line(size=1),   ###坐标轴的刻度粗细
legend.key = element_rect(fill="white"),
#legend.position=c(1.179,0.5),  ###设置图例的位置
#legend.key.size = unit("0.3","cm")
#legend.background = element_blank(colour="None",fill="None"),
plot.margin=unit(rep(1,4),'lines'),
)

###图2
dat <- read.csv("C:/Users/Lenovo/Desktop/Bioinformatics-master/plot/scatter/L1_L2.csv")
data1 <- data.frame(dat)
###> coef(lm(Negative ~ Positive, data = dat))###计算最佳拟合直线的斜率和直径
###  (Intercept)    Positive 
###  0.1935481   0.4202044 
data1.lm <- lm(Negative ~ Positive, data = data1)
p2 <- ggplot(data1,aes(x = Positive,y = Negative,colour=L_length,shape=value))+
geom_point(alpha = 0.7,size=3)+
facet_wrap(~L_length)+
guides(colour = "none")+
geom_abline(intercept = coef(data1.lm)[1],slope=coef(data1.lm)[2],linetype="longdash")+
labs(tag = "B",x="The value of Positive Samples",y="The value of Negative Samples")+
theme(
axis.text = element_text(colour="black"), ###设置坐标轴文本的大小
axis.title = element_text(size = rel(1.2),colour="black"), ###设置坐标轴标签的大小
panel.background = element_rect(fill = "white", colour = "black"),###将面板背景变空白
panel.grid.major = element_blank(),  
panel.grid.minor = element_blank(),
axis.title.x = element_text(vjust=-1), ###x轴标签距离x轴的大小
axis.title.y = element_text(vjust=4), ###y轴标签距离y轴的大小
axis.ticks = element_line(size=1),   ###坐标轴的刻度粗细
legend.key = element_rect(fill="white"),
#legend.position=c(1.179,0.5),  ###设置图例的位置
#legend.key.size = unit("0.3","cm")
#legend.background = element_blank(colour="None",fill="None"),
plot.margin=unit(rep(1,4),'lines'),
)
###显示图片
vplayout <- function(x,y){viewport(layout.pos.row = x, layout.pos.col = y)} 
grid.newpage()  ###新建图表版面
pushViewport(viewport(layout = grid.layout(2,1)))
print(p1, vp = vplayout(1,1)) ###将(2,1)的位置画图p1
print(p2, vp = vplayout(2,1)) ###将(2,1)的位置画图p1

###保存图片
pdf("C:/Users/Lenovo/Desktop/Bioinformatics-master/plot/scatter/scatter.pdf",width=4,height=8)### 保存图片
vplayout <- function(x,y){viewport(layout.pos.row = x, layout.pos.col = y)}
grid.newpage()  ###新建图表版面
pushViewport(viewport(layout = grid.layout(2,1)))
print(p1, vp = vplayout(1,1)) ###将(2,1)的位置画图p1
print(p2, vp = vplayout(2,1)) ###将(2,1)的位置画图p1
dev.off()
