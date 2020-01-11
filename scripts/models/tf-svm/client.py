#coding=utf8
'''
Created on 2018年10月17日

@author: 95890
'''

"""Send text to tensorflow serving and gets result
"""


import data_helpers
import numpy as np
import sys
from data_transformer import DataTransformer
import requests
import json

train_data_file = './data/sample.csv'
def main():
  # Send request
    # See prediction_service.proto for gRPC request/response details.
  X =["@kg.MutualFund 基金@初始规模 是怎样"] # expected output: y = 1
  y =["class_id_1"]
  data_transformer = DataTransformer(train_data_file)
  X_encoded, y_encoded = data_transformer.fit(X, y);

  # REST
  url = sys.argv[1]
  data = {'signature_name':'textclassified','instances':[{'inputX': X_encoded[0].tolist() }]}
  data = json.dumps(data)
  r = requests.post(url, data=data)
  print('test:{}'.format(X[0]))
  print(r.text)
  print('y_true:{}'.format(y_encoded))


if __name__ == '__main__':
  main()
