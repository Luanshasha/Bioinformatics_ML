Explain:
====
论文：Wei, L., et al., Iterative feature representations improve N4-methylcytosine site prediction. Bioinformatics, 2019.<br><br>
1. 运行代码：`ItSVM.py`:<br>
    python ItSVM.py -i input.fasta -g gen_type -f fill_NA -t iter_times -n cv_number -c CPU_value -2rfh  2RFH_distance -ath AthMethPre_distance -knn KNN_distance -mmi MMI_distance -pcp PCP_distance -pd PseDNC_distance -pe PseEIIP_distance -rfh RFH_distance<br><br>
    * input.fasta输入是fasta格式文件。<br>
    * gen_type代表输入序列是DNA还是RNA，若是DNA，gen_type就为DNA。<br>
    * fill_NA 代表是否有N填充,此处可以输入 0 或者 1, 0代表有N填充，1相反。<br>
    * iter_times代表8种概率特征融合之后跑迭代的次数，本次论文设置的是60。<br>
    * cv_number是几折交叉验证，2RFH_distance代表2RFH.py特征提取程序跑出来的特征，使用特征优化程序时步长是多少：默认值是5，代表步长是5。<br>
    * CPU_value代表CPU运行的核数，windows系统一般输入3，服务器一般输入8。<br>
    * -2rfh 2RFH_distanc：前面代表的是2RFH.py特征提取程序跑出来的特征，2RFH_distanc代表是提取出来的特征跑特征优化（SVM_distance.py文件可见）时的步长。后面的依次类推。总共8种特征提取方法---features_extract.py文件中可见。<br>
    * cv_number,2RFH_distance,AthMethPre_distance,KNN_distance,MMI_distance,PCP_distance,PseDNC_distance,PseEIIP_distance,RFH_distance默认值分别是10，5，10，2，1，5，1，4，5，这些带有默认值的参数，若是不输入自己的参数，就采用默认值；若是输入自己的参数，就使用输入的参数。<br><br>
    
    
    examples:<br>
    全部参数都设置：<br>
    python ItSVM.py -i p_n_seq.fasta -g RNA -f 0 -t 60 -n 5 -c 3 -2rfh 20 -ath 50 -knn 30 -mmi 20 -pcp 20 -pd 40 -pe 5 -rfh 10<br>
    部分参数设置（采用默认值值）：<br>
    python ItSVM.py -i p_n_seq.fasta -g RNA -f 0 -t 60 -c 3 


