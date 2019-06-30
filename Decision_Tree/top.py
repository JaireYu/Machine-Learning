#encoding=utf-8
import TreePlotter
import Decide_Tree_library
fr = open("lenses.txt")
lenses = [line.strip().split("\t") for line in fr.readlines()]      #readlines返回的是每一个列表
#变量列表元素，用strip形成新的列表作为主列表元素
lensesLabels = ["age", "prescript", "astigmatic", "tearrate"]
lensesTree = Decide_Tree_library.createTree(lenses, lensesLabels)
print lensesTree
TreePlotter.createPlot(lensesTree)