# encoding=utf-8
import hiai

class ModelManager(object):
    def __init__(self):
        pass

    def CreateGraph(self, model,graph_id, model_engine_id):
        myGraph = hiai.Graph(hiai.GraphConfig(graph_id = graph_id))
        if myGraph is None:
            print('get graph failed')
            return None
        with myGraph.as_default():
            model_engine = hiai.Engine(hiai.EngineConfig(engine_name='ModelInferenceEngine', side=hiai.HiaiPythonSide.Device,engine_id = model_engine_id))
            if model_engine is None:
                print('get model_engine failed')
                return None
            else:
                print('get model_engine ok!')
            with model_engine.as_default():
                if (None == model_engine.inference(input_tensor_list=hiai.NNTensorList(), ai_model=model)):
                    print('Init model_engine failed ')
                    return None
                else:
                    print('Init model_engine ok!')
        # Create Graph
        if (hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
            print('create graph ok ')
            return myGraph
        else:
            print('create graph failed')
            return None


    def Inference(self, graphHandle, inputTensorList):
        if not isinstance(graphHandle, hiai.Graph):
            print("graphHandle is not Graph object")
            return None
        resultList = graphHandle.proc(inputTensorList)
        if resultList is None:
            print('Inference error')
        return resultList
