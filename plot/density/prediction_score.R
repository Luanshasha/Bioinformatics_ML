###S3_prediction_score
library(ggplot2)
library(grid)

###图1
test <- read.csv("C:/Users/Lenovo/Desktop/R/S3_presion.csv")
data1 <- data.frame(labels = test[,1],scores=test[,2])
p1 <- ggplot(data1 ,aes(x=scores,fill=labels))+
geom_density(alpha=.3)+labs(tag = "A",y="Density",x="Prediction scores")+
guides(fill = guide_legend(title = NULL,reverse=TRUE))+ ###去掉图标的头文本，将图例的进行翻转
theme(axis.text = element_text(size = rel(1.2),colour="black"), ###设置坐标轴文本的大小
axis.title = element_text(size = rel(1.2),colour="black"), ###设置坐标轴标签的大小
panel.background = element_rect(fill = "white", colour = "black"),
#panel.border = element_rect(color="black"),
panel.grid.major = element_blank(),  ###将面板背景变空白
panel.grid.minor = element_blank(),
legend.text=element_text(size=10),
#axis.line=element_line(color="black",size=1),
legend.position=c(0.52,0.86),   ###图标的位置
axis.title.x = element_text(vjust=-1), ###x轴标签距离x轴的大小
axis.title.y = element_text(vjust=5), ###y轴标签距离y轴的大小
axis.ticks = element_line(size=1),   ###坐标轴的刻度粗细
)


###图2
test <- read.csv("C:/Users/Lenovo/Desktop/R/S3_presion.csv")
data2 <- data.frame(labels = test[,1],scores=test[,2])
p2 <- ggplot(data2 ,aes(x=scores,fill=labels))+
geom_density(alpha=.3)+labs(tag = "B",y="Density",x="Prediction scores")+
guides(fill = guide_legend(title = NULL,reverse=TRUE))+ ###去掉图标的头文本，将图例的进行翻转
theme(axis.text = element_text(size = rel(1.2),colour="black"), ###设置坐标轴文本的大小
axis.title = element_text(size = rel(1.2),colour="black"), ###设置坐标轴标签的大小
panel.background = element_rect(fill = "white", colour = "black"),
#panel.border = element_rect(color="black"),
panel.grid.major = element_blank(),  ###将面板背景变空白
panel.grid.minor = element_blank(),
legend.text=element_text(size=10),
#axis.line=element_line(color="black",size=1),
legend.position=c(0.52,0.86),   ###图标的位置
axis.title.x = element_text(vjust=-1), ###x轴标签距离x轴的大小
axis.title.y = element_text(vjust=5), ###y轴标签距离y轴的大小
axis.ticks = element_line(size=1),   ###坐标轴的刻度粗细
)
###显示图片
vplayout <- function(x,y){viewport(layout.pos.row = x, layout.pos.col = y)} 
grid.newpage()  ###新建图表版面
pushViewport(viewport(layout = grid.layout(2,1)))
print(p1, vp = vplayout(1,1)) ###将(2,1)的位置画图p1
print(p2, vp = vplayout(2,1)) ###将(2,1)的位置画图p1

###保存图片
pdf("C:/Users/Lenovo/Desktop/R/density16.pdf",width=4,height=8)### 保存图片
vplayout <- function(x,y){viewport(layout.pos.row = x, layout.pos.col = y)}
grid.newpage()  ###新建图表版面
pushViewport(viewport(layout = grid.layout(2,1)))
print(p1, vp = vplayout(1,1)) ###将(2,1)的位置画图p1
print(p2, vp = vplayout(2,1)) ###将(2,1)的位置画图p1
dev.off()