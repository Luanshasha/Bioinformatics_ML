# Explain
此文件夹存放的是特征提取程序。<br><br>
* `MMI.py`文件运行命令：<br>
>>>>python program.py input.fasta output.csv<br><br>
        
example: <br>
>>>>python MMI.py 4mC.fasta 4mC_MMI.csv<br><br>

* `k-mer.py`文件运行命令：<br>
>>>>python program.py input.fasta output.csv gen_type fill_NA k<br><br>
        
>>>>>>>注：<br>
>>>>>>>* gen_type 为 DNA　或者　RNA<br>
>>>>>>>* fill_NA　为　0　或者　1. 0 代表没有　N　填充，1 代表有　N　填充<br>
>>>>>>>* k　代表　k-mer的取值，1-6 左右，数值太大运行很慢<br><br>

example：<br>
>>>>python k-mer.py 4mC.fasta 4mC_4mer.csv DNA 0 4<br><br>
        
* `binary-profile-base-features.py`文件运行命令：<br>
>>>>python program.py input.fasta output.csv gen_type fill_NA<br><br>

example：<br>
>>>>python binary-profile-base-features.py 4mC.fasta 4mC_binary.csv DNA 0<br><br>

    　
