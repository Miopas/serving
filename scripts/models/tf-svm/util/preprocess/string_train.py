#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''字符串预处理,去除停用词.

    normalize:去除停用词
    normalize_with_num:去除停用词并将数字转为'NUMBER'
'''

import re

def normalize(s):
    tmp = s.strip()
    tmp = re.sub(ur'^http.*$', u'', tmp)
    tmp = re.sub(ur'[ 、，。？：（）【】〜！/\@\-\",:<>~“”\'\(\)\[\]⋯?\$\!\\\.\_\^]', u'', tmp)
    tmp = re.sub(ur'^[a-zA-Z0-9\.\*\+\-\_]+$', u'', tmp)
    return tmp.lower()

def normalize_with_num(s):
    tmp = normalize(s)

    tmp = re.sub(ur'[0-9]+', u'NUMBER', tmp)
    return tmp

