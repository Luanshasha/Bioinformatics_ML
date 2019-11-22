@echo off
python RFH.py NS_628.txt NS_628_RFH.csv RNA 0
python PseEIIP.py NS_628.txt NS_628_PseEIIP.csv RNA 0
python PseDNC.py NS_628.txt NS_628_PseDNC.csv RNA 0 
python PCP.py NS_628.txt NS_628_PCP.csv RNA 0
python MMI.py NS_628.txt NS_628_MMI.csv RNA 0
python KNN.py NS_628.txt NS_628_KNN.csv RNA 0
python AthMethPre.py NS_628.txt NS_628_AthMethPre.csv RNA 0
python 2RFH.py NS_628.txt NS_628_2RFH.csv RNA 0
Pause