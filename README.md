環境:WIN10 Python 3.7.4

mdp_Cohort_4.py是修改公式後的程式

Line 10:每個Sensor的能耗，分別是clock、light、bed、card、TV的能耗

Line 18:方法B與方法A切換，True:方法B False:方法A

執行指令python mdp_Cohort_4.py 0.9 0.7

第一個參數:0.9是 MDP衰減率

第二個參數:0.7 是Sensor的r值

ontology:

![image](https://github.com/ice71115/MDP/blob/master/image/ontology.png)

轉移機率更新頻率 : 86400秒更新一次

CASE1: 能耗

sClock sLight sBed sCard sTV

  6       3     2     3   2.5
  
CASE2: 能耗

sClock sLight sBed sCard sTV

  6       3     6     3   2.5
  
CASE3: 能耗

sClock sLight sBed sCard sTV

  2       3     6     3   2.5


Case1結果:

![image](https://github.com/ice71115/MDP/blob/master/Results/case1/Case1.png)

Case2結果:

![image](https://github.com/ice71115/MDP/blob/master/Results/case2/Case2.png)

Case3結果:

![image](https://github.com/ice71115/MDP/blob/master/Results/case3/Case3.png)

