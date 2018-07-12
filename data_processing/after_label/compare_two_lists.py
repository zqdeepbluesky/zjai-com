# -*- coding: utf-8 -*-
# @Time    : 5/22/2018
# @Author  : CarrieChen
# @File    : compare_two_lists.py
# @Software: ZJ_AI
#this code is for compare two lists in excel


import xlrd

list1=[]
list2=[]

data2 = xlrd.open_workbook('C:\\Users\\Administrator\\Desktop\\data_processing_carriechen\\work_folder\\count_Annotations.xls')
table2 = data2.sheets()[0]
list1=table2.col_values(0)
list2 = table2.col_values(1)


for e in range(1,len(list1)):
    if list1[e]!=list2[e]:
        print("No:"+str(e+1)+"   "+"wrong:"+list1[e]+"  right:"+list2[e])
print("success")
