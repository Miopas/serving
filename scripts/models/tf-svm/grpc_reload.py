#coding=utf-8
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2
from tensorflow_serving.util import status_pb2

import grpc
import sys

def run():
  channel = grpc.insecure_channel(sys.argv[1])
  stub = model_service_pb2_grpc.ModelServiceStub(channel)
  request = model_management_pb2.ReloadConfigRequest()  ##message ReloadConfigRequest
  model_server_config = model_server_config_pb2.ModelServerConfig()

  config_list = model_server_config_pb2.ModelConfigList()##message ModelConfigList

  one_config = config_list.config.add() #####try to add one model config
  one_config.name= "svm_cls"
  one_config.base_path = "/models/svm_cls"
  one_config.model_platform="tensorflow"
  #one_config.model_version_policy.specific.versions.append(4)


  model_server_config.model_config_list.CopyFrom(config_list)
  request.config.CopyFrom(model_server_config)

  #print(request.IsInitialized())
  #print(request.ListFields())

  responese = stub.HandleReloadConfigRequest(request,10)
  if responese.status.error_code == 0:
      print("Reload sucessfully")
  else:
      print("Reload failed!")
      print(responese.status.error_code)
      print(responese.status.error_message)

run()
