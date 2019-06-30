#coding=utf-8
import matplotlib.pyplot as plt

#创建参数字典
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8") #节点类型锯齿形和边线0.8
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")    #创建参数字典

def PlotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = "axes fraction", xytext = centerPt,
                            textcoords = "axes fraction", va = "center", ha = "center", bbox = nodeType,
                            arrowprops = arrow_args)

def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)   #函数句柄当成局部变量看？
    PlotTree.totalW = float(GetNumLeafs(inTree))    #全局变量存储总宽度
    PlotTree.totalID = float(GetTreeDepth(inTree))  #全局变量存储深度
    PlotTree.xOff = -0.5/PlotTree.totalW    #初始时放在-0.5处因为在plotTree中会+1
    PlotTree.yOff = 1.0
    PlotTree(inTree, (0.5, 1.0), "")
    plt.show()

def GetNumLeafs(MyTree):    #计算叶子节点个数
    NumLeafs = 0
    FirstStr = MyTree.keys()[0]
    SecondDict = MyTree[FirstStr]
    for key in SecondDict.keys():
        if(type(SecondDict[key]).__name__ == "dict"):
            NumLeafs += GetNumLeafs(SecondDict[key])
        else:
            NumLeafs += 1
    return NumLeafs

def GetTreeDepth(MyTree):   #计算深度
    if(type(MyTree).__name__ != "dict"):
        return 1
    maxDepth = 0
    FirstStr = MyTree.keys()[0]
    SecondDict = MyTree[FirstStr]
    for key in SecondDict.keys():
        if(GetTreeDepth(SecondDict[key]) + 1 > maxDepth):
            maxDepth = GetTreeDepth(SecondDict[key]) + 1
    return maxDepth

def PlotMidText(CntrPt, ParentPt, txtString):   #在父子之间填充文本
    xMid = (ParentPt[0] - CntrPt[0])/2.0 + CntrPt[0]
    yMid = (ParentPt[1] - CntrPt[1])/2.0 + CntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def PlotTree(MyTree, ParentPt, NodeTxt):    #给出父节点位置和节点文本画树
    NumLeafs = GetNumLeafs(MyTree)
    Depth = GetTreeDepth(MyTree)
    FirstStr = MyTree.keys()[0]
    CntrPt = (PlotTree.xOff + (1.0 + float(NumLeafs))/2.0/PlotTree.totalW, PlotTree.yOff) #获取父节点的xy坐标：为什么/2.0？定位到中间
    PlotMidText(CntrPt, ParentPt, NodeTxt)
    PlotNode(FirstStr, CntrPt, ParentPt, decisionNode)
    SecondDict = MyTree[FirstStr]
    PlotTree.yOff = PlotTree.yOff - 1.0/PlotTree.totalID
    for key in SecondDict.keys():
        if(type(SecondDict[key]).__name__ == "dict"): #如果是dict
            PlotTree(SecondDict[key], CntrPt, str(key))
        else:
            PlotTree.xOff = PlotTree.xOff + 1.0/PlotTree.totalW #计算子节点在同一层只需迭代x坐标
            PlotNode(SecondDict[key], (PlotTree.xOff, PlotTree.yOff), CntrPt, leafNode)
            PlotMidText((PlotTree.xOff, PlotTree.yOff), CntrPt, str(key))
    PlotTree.yOff = PlotTree.yOff + 1.0/PlotTree.totalID #回到上一层




