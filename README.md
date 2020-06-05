環境:Python 3.7.4

Line 10:每個Sensor的能耗，分別是clock、light、bed、card、TV的能耗

Line 18:方法B與方法A切換，True:方法B False:方法A

 ![image](https://github.com/ice71115/MDP/blob/master/image/instruction.png)
 
第一個參數:0.9是 MDP衰減率
第二個參數:0.7 是Sensor的r值

ontology:

![image](https://github.com/ice71115/MDP/blob/master/image/ontology.png)

轉移機率更新頻率 : 86400秒更新一次
