class Queue(object):# 定义队列类 为什么加入object
    def __init__(self,size):
       self.size = size #定义队列长度
       self.queue = []#存储队列 列表
    #返回对象的字符串表达式 方便调试
    def __str__(self):
        return str(self.queue)#什么含义

    #初始化队列
    #def init(self):
    #入队
    def inQueue(self,n):
        if self.isFull():
            return -1
        self.queue.append(n)#列表末尾添加新的对象
    #出队
    def outQueue(self):
        if self.isEmpty():
            return -1
        firstElement = self.queue[0]  #删除队头元素
        self.queue.remove(firstElement) #删除队操作
        return firstElement
    #删除某元素
    def delete(self,n,m):
        self.queue[n] = m
    #插入某元素
    def inPut(self,n,m):#n代表列表当前的第n位元素 m代表传入的值
        self.queue[n] = m
    #获取当前长度
    def getSize(self):
        return len(self.queue)
    #判空
    def isEmpty(self):
        if len(self.queue)==0:
            return True
        return False
    #判满
    def isFull(self):
        if len(self.queue) == self.size:
            return True
        return False

'''
#if __name__ == '__main__':#如何使用?
queueTest = Queue(10)
for i in range(10):
    queueTest.inQueue(i)
print('列表元素',queueTest)
print('列表长度',queueTest.getSize())
print('判空',queueTest.isEmpty())
print('判满',queueTest.isFull())
queueTest.inPut(1,20)#list【0】 = 2
queueTest.outQueue()#出队
print('当前列表',queueTest)
"""
结果：
列表元素 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
列表长度 10
判空 False
判满 True
当前列表 [20, 2, 3, 4, 5, 6, 7, 8, 9]
"""
'''