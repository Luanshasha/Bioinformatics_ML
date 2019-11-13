# Explain
此文件夹存放的是特征提取程序。<br><br>
1. `MMI.py`文件运行命令：<br>
>>>>python program.py input.fasta output.csv<br>
        
example: <br>
>>>>python MMI.py 4mC.fasta 4mC_MMI.csv<br><br>

2. `k-mer.py`文件运行命令：<br>
>>>>python program.py input.fasta output.csv gen_type fill_NA k<br>
        
>>>>>>注：<br>
>>>>>>* gen_type为 DNA 或者 RNA。<br>
>>>>>>* fill_NA为 0 或者 1。 0代表没有 N 填充，1 代表有 N 填充。<br>
>>>>>>* k　代表k-mer的取值，1-6 左右，数值太大运行很慢。<br>

example：<br>
>>>>python k-mer.py 4mC.fasta 4mC_4mer.csv DNA 0 4<br><br>
        
3. `binary-profile-base-features.py`文件运行命令：<br>
>>>>python program.py input.fasta output.csv gen_type fill_NA<br>

example：<br>
>>>>python binary-profile-base-features.py 4mC.fasta 4mC_binary.csv DNA 0<br><br>

    　
