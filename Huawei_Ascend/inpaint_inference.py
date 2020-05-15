# encoding=utf-8
import ModelManager
import hiai
from hiai.nn_tensor_lib import DataType
import numpy as np
import argparse
import time
import pdb


class InpaintInference(object):
    def __init__(self,model_path):
        # set graph id for inpaint inference model
        self.graph_id = 1000
        self.model_engine_id = 100
        self.model = ModelManager.ModelManager()
        self.graph = None
        self.model_path = model_path
        self._getgraph()

    def __del__(self):
        self.graph.destroy()

    def _getgraph(self):
        # description of model
        inferenceModel = hiai.AIModelDescription('inpaint', self.model_path)
        # init graph
        self.graph = self.model.CreateGraph(inferenceModel,self.graph_id,self.model_engine_id)
        if self.graph is None:
            print("Init Graph failed")

    def Inference(self,img,mask):
        input_tensor_list = []
        sizes = [3*512*512,512*512]
        input_tensor_list.append(hiai.NNTensor(img, height=512, width = 512, channel = 3 ,name = 'image', data_type = DataType.FLOAT32_T, size = sizes[0]))
        input_tensor_list.append(hiai.NNTensor(mask, height=512 , width = 512, channel = 1,name = 'mask', data_type = DataType.FLOAT32_T, size = sizes[1]))
        nntensorList=hiai.NNTensorList([input_tensor_list[0],input_tensor_list[1]])
        if not nntensorList:			
            print("nntensorList is null")
            
        resultList = self.model.Inference(self.graph, nntensorList)

        inpainted_512 = resultList[0]
        attention = resultList[1]
        mask_512_new = resultList[2]

        # output in memory is NCHW, but in numpy it is displayed as NHWC, need to make numpy consistent with memory
        inpainted_512.shape = (1,3,512,512)
        attention.shape = (1,1024,32,32)
        mask_512_new.shape = (1,1,512,512)

        # transpose to NHWC
        inpainted_512 = inpainted_512.transpose(0,2,3,1)
        attention = attention.transpose(0,2,3,1)
        mask_512_new = mask_512_new.transpose(0,2,3,1)

        return inpainted_512,attention,mask_512_new

if __name__ == '__main__':
    
    #prepare binary files 'img_1.chw' and 'mask_1.chw' to test the model itself
    parser = argparse.ArgumentParser(description="Test om model accuracy")
    parser.add_argument('--model', type=str, default='./inpaint.om')
    parser.add_argument("-i", "--input_bins", help="input_bins bins. e.g. './a.bin;./c.bin'", default = 'img_1.chw;mask_1.chw')
        
    args = parser.parse_args()
    model = args.model
    input_bins = args.input_bins.split(';')

    img = np.fromfile(input_bins[0],np.float32).reshape([3,512,512]).copy()
    mask = np.fromfile(input_bins[1],np.float32).reshape([512,512]).copy()

    inpaint_app = InpaintInference(model)

    inpainted, attention, mask_processed = inpaint_app.Inference(img,mask)

    inpainted.tofile('inpained.bin')
    attention.tofile('attention.bin')
    mask_processed.tofile('mask_processed.bin')













