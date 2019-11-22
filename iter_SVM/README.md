Explain:
====
此处两个文件分别是对组合后的概率特征进行迭代和迭代之后的独立测试。具体请看论文：Wei, L., et al., Iterative feature representations improve N4-methylcytosine site prediction. Bioinformatics, 2019.<br><br>
1. `ItSVM.py`:<br>
    python ItSVM.py -i input.data -t iter_times -n CV_numbers -c CPU_value<br><br>
    examples:<br>
    python ItSVM.py -i probability_features.csv -t 60 -n 10 -c 3<br><br>
2. `ItSVM_test.py`:<br>
    python ItSVM.py -i input.data -t iter_times <br><br>
    examples:<br>
    python ItSVM_test.py -i Test.csv -t 60 <br>


