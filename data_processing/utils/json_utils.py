#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/26/2018 10:36 AM 
# @Author : sunyonghai 
# @File : json_utils.py 
# @Software: ZJ_AI
# =========================================================
import json
data = {'name': 'Firmin',
        'age': 25,
        'sex':'men'
        }

json_str = json.dumps(data)
print(type(json_str))