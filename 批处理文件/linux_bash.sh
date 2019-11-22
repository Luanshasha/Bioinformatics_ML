#!bin/bash
source activate py3;
cd p_n_2RFH;
python SVM_distance.py p_n_seq_2RFH.csv 3 10 8;
cd ..;
cd p_n_AthMethPre;
python SVM_distance.py p_n_seq_AthMethPre.csv 5 10 8;
cd ..;
cd p_n_KNN;
python SVM_distance.py p_n_seq_KNN.csv 4 10 8;
cd ..;
cd p_n_MMI;
python SVM_distance.py p_n_seq_MMI.csv 2 10 8;