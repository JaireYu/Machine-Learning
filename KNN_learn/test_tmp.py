import kNN
import imgProcess
while(1):
    Str = raw_input("input a hand writing picture:\n")
    kNN.hwTest(imgProcess.picTo01("D:\\python_learning\\KNN_learn\\{}".format(Str)))
