#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''提供预测时的字符串预处理器

    normalize:去除停用词
    normalize_with_num:去除停用词并将数字转为'NUMBER' 
'''

import re

def normalize(s):
    tmp = s.strip()
    tmp = re.sub(ur'(([^@]|^)基金)公司', ur'\1', tmp)
    tmp = re.sub(ur'(产品代码：|产品代码:|@询问 |@描述 )', u'', tmp)
    tmp = re.sub(ur'^http.*$', u'', tmp)
    tmp = re.sub(ur'[ 、，。？：（）【】〜！/\@\-\",:<>~“”\'\(\)\[\]⋯?\$\!\\\.\_\^]', u'', tmp)
    tmp = re.sub(ur'^[a-zA-Z0-9\.\*\+\-\_]+$', u'', tmp)
    tmp = tmp.lower()
    tmp_back = tmp
    tmp = re.sub(ur'(你那有这个信息|截止到现在为止|截止到当前为止|还想了解一下|hello|再帮忙查看|想了解一下|截止到现在|请发我一下|截止到目前|截止到现在|截止到当前|也告诉我下|先了解一下|前一段时间|前段时间)', u'', tmp)
    tmp = re.sub(ur'(在职期间|我不记得|请发一下|你能看到|买了以后|我购买了|大概简单|介绍一下|提供一下|数据库里|想咨询下|告诉我下)', u'', tmp)
    tmp = re.sub(ur'(有没有|告诉下|了解下|新发行|数据库|想问问|再查下|你可以|告诉我|发一下|想问下|你知道|能不能|这一只)', u'', tmp)
    tmp = re.sub(ur'(这一支|这一款|这一个|有的话|你清楚|麻烦你|在你们|我想买|不知道|从长远|发给我|确认下)', u'', tmp)
    tmp = re.sub(ur'(上个月|上一周|上星期|前几天|前两周|前两天|前一天|不是很|参考下)', u'', tmp)
    tmp = re.sub(ur'(帮看看|你那里|我如果|咨询|一下|情况|谢谢|请问|另外|好的|麻烦|你好|您好|哈喽|hi|能否|准备|需要|看下|顺便|提供|清楚)', u'', tmp)
    tmp = re.sub(ur'(明白|不太|理解|持有|准备|详细|仔细|在哪|查到|参考|发送|过来|如何|具体)', u'', tmp)
    tmp = re.sub(ur'(一般|觉得|你能|你们|给我|了解|问下|问问|查下|最近|现在|当前|目前|前天|昨天|今天|明天|后天|打算|我想|可以|最后|之后|哪里|如果)', u'', tmp)
    tmp = re.sub(ur'(还有|办理|这支|帮我|非常|感谢|谢啦|谢了|请你|请教|有吗|这家|这个|这款|此款|该款|这支|这只|大概|到时|帮忙|确认)', u'', tmp)
    tmp = re.sub(ur'(ok|也|把|帮|还|我|要|道|嗯|哈|吗|懂|呢|嘞|那|看|又|能|的|得|地|再|咧|该|但|想|请|瞅|瞧|另|此|说|侃|噢|哇|恩|呐|哦|耶)', u'', tmp)
    tmp = re.sub(ur'(呀|嘛|诶|啦|喂|切|咧|咦|呃|噫|咿|撒|哒|伐|咯|噶|蛤|哩|嘻|吧|咩)', u'', tmp)
    tmp = re.sub(ur'[0-9一二三四五六七八九十](个月|个亿|亿|年|月)', u'', tmp)
    tmp = re.sub(ur'[0-9]', u'', tmp)
    #tmp = re.sub(ur'([^\d])([0-9]{1,4}|[0-9]{8,})([^\d])', u'\1\3', tmp)
    tmp = tmp.strip()
    back_list = [u'kgmutualfund',u'kgmutualfundmanager',u'kgmutualfundcompany',u'kgmutualfund基金',u'kgmutualfund产品',u'kgmutualfundmanager人', u'kgmutualfundmanager基金经理',u'kgmutualfundcompany公司',u'kgmutualfundcompany基金公司',u'它',u'他',u'她',u'产品',u'基金',u'基金经理',u'基金公司']
    #print tmp
    if tmp == u'' or tmp in back_list:
        tmp = tmp_back
    #print tmp
    return tmp

def normalize_with_num(s):
    tmp = normalize(s)

    tmp = re.sub(ur'[0-9]+', u'NUMBER', tmp)
    return tmp
