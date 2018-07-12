# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys

def read_all_keys():
    allkey=[]
    f=open(r"sum_Annotations.txt","r")
    line=f.readline()
    while line:
        aftersplit1=line.split(":",1)[0] #key
        allkey.append(aftersplit1)
        line=f.readline()
    f.close()
    return unique(allkey)
     
    
def unique(list):  
    newlist = []  
    for x in list:  
        if x not in newlist:  
            newlist.append(x)  
    return newlist  
    
def append_values(sum_all):
     f=open(r"sum_Annotations.txt","r")
     line=f.readline()
     while line:
         key=line.split(":",1)[0] #key
         value=line.split(":",1)[1]
         value=value.split("\n",1)[0] #value
         sum_all[key]=int(sum_all[key])+int(value)
         line=f.readline()
     f.close()
     return sum_all

if __name__ == "__main__":
    keylist=read_all_keys()
    sum_all={}
    for i in keylist:
        sum_all[i]='0'
    sum_all=append_values(sum_all)
    print(sum_all)