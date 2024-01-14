train.txt和test.txt为最原始的实验提供数据文本。
截图证据.pdf为实验过程中大部分测试的截图，由于放进实验报告中过于冗杂不利于排版和美观，故单独放出。
数据表.xlsx为统合的数据表格，便于直接对比分析。

代码文件夹中，split.py为将原始数据文本随机按照训练集：测试集等于8:2划分的程序。在我实验过程中已经随机生成了三组数据，分别为train_split1.txt，test_split1.txt和train_split2.txt，test_split2.txt以及train_split3.txt，test_split3.txt.全存放于文本集文件夹中。

文本集文件夹中存放了停用词表ban.txt，实验提供的初始数据train.txt和test.txt。以及经过split.py随机生成的三组数据文本,还有我手动将原始数据集依顺序按照8:2划分的train_split.txt,test_split.txt。