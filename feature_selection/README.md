# Explain
此文件夹存放的是特征提取程序。<br><br>
MMI.py文件运行命令：<br>
        python program.py input.fasta output.csv<br><br>
        
example: <br>
        python MMI.py 4mC.fasta 4mC_MMI.csv<br><br>

k-mer.py文件运行命令：<br>
        python program.py input.fasta output.csv gen_type fill_NA k<br><br>
        
  注：1. gen_type 为 DNA　或者　RNA<br>
  　　２.　fill_NA　为　０　或者　１。０代表没有　N　填充，１代表有　Ｎ　填充<br>
    　３.　k　代表　k-mer的取值，１－６左右，数值太大运行很慢<br><br>

example：<br>
        python k-mer.py 4mC.fasta 4mC_4mer.csv DNA 0 4<br><br>
        
binary-profile-base-features.py文件运行命令：<br>
         python program.py input.fasta output.csv gen_type fill_NA<br><br>
         
example：<br>
        python binary-profile-base-features.py 4mC.fasta 4mC_binary.csv DNA 0<br>

    　
