# encoding=utf-8
import ModelManager
import hiai
from hiai.nn_tensor_lib import DataType
import numpy as np
import time
import argparse

class MatMulInference(object):
    def __init__(self,model_path):
        self.graph_id = 1001
        self.model_engine_id = 100
        self.model = ModelManager.ModelManager()
        self.graph = None
        self.model_path = model_path
        self._getgraph()

    def __del__(self):
        self.graph.destroy()

    def _getgraph(self):
        inferenceModel = hiai.AIModelDescription('matmul', self.model_path)
        # init Graph
        self.graph = self.model.CreateGraph(inferenceModel,self.graph_id,self.model_engine_id)
        if self.graph is None:
            print("Init Graph failed")


    def Inference(self,x,y):
        x = x.reshape(1024,1024).copy() #make sure memory order is the same as np order
        y = y.reshape(1024,3072).copy() #make sure memory order is the same as np order
        sizes = [1024*1024,1024*3072]
        input_tensor_list = []
        input_tensor_list.append(hiai.NNTensor(x, height=1024, width = 1024, channel = 1 ,name = 'x', data_type = DataType.FLOAT32_T, size = sizes[0]))
        input_tensor_list.append(hiai.NNTensor(y, height=1024 , width = 3072, channel =1,name = 'y', data_type = DataType.FLOAT32_T, size = sizes[1]))
        nntensorList=hiai.NNTensorList([input_tensor_list[0],input_tensor_list[1]])
        if not nntensorList:			
            print(" matmul nntensorList is null")
        resultList = self.model.Inference(self.graph,nntensorList)
        # pdb.set_tracce()
        result = resultList[0]
        return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test matmul om model accuracy")
    parser.add_argument('--model', type=str, default='./matmul.om')
        
    args = parser.parse_args()
    model = args.model

    x = np.random.random(
        1024*1024).astype(np.float32).reshape(1024, 1024)
    y = np.random.random(
        1024*3072).astype(np.float32).reshape(1024,3072)

    matmul_app = MatMulInference(model)
    
    result = matmul_app.Inference(x,y)
    
    print(result)













