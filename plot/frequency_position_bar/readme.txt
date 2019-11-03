说明：
     1.将S4.txt数据集分成两份，一份是正例数据集S4_pos.txt, 一份是负例数据集S4_neg.txt。
     2.用frequency_position.py程序分别对这两个数据集提取，得到输出文S4_pos.csv和S4_neg.csv。
       提取命令：
                 python frequency_position.py input.txt output.csv
        Example:
	         python frequency_position.py S4_pos.txt S4_pos.csv
     3.使用jupyter打开frequency_position_bar.ipynb，进行画图。将S4_pos.csv文件和S4_neg.csv文件传到相应的位置即可。