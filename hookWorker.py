import sys
from torch.autograd import Function

import torch.nn.functional as F
###########################################################
sys.path.append('G:\AI/your project main path/python2ncnn') ###Must changed for every project ,all locate pybridge package to fixed-location in Windows
###########################################################
from torchLayer_Interpreter import Layer_Interpreter,TorchLayers_Keeper
from torchLayer_ncnn import NCNN_FRAMEWORK_FACTORY
from torchLayer_Graph import TorchGraph_Pen
######################################################################
old_function__call__ = Function.__call__
global_hooks_handles = []
##will be supported later,maybe...
FRAME_WORK_TYPE = {
    "DEFAULT"   :   0,
    "CAFFE"     :   1,
    "TENSOR_FLOW":  2,
    "ONNX"       :  3
}

def module_forward_hooking(module,inputList, output):
    mdLayer = Layer_Interpreter(module,inputList, output)
    res       = mdLayer.Layers_Interpretering(None,None)
    if res:
        print("module:", {id(module): str(module.__class__)})
    else:
        print("*******your layers in model , something is Null or can not be interpretered...")

    ####################do some graph vision op later##############
    return True

def funcs_forward_hooking():
    res = Layer_Interpreter.funcs_Interpretering()
    return None


def funcsParams_hook():
    funcs_forward_hooking()
    print("funcs here...")

def forward_hooks_mount2_Net(netModel):
    inum = 0
    def moduleParams_hook(netModule, inputs, output):
        inLists = list(inputs)  # listed for tensor for hooks
        module_forward_hooking(netModule, inLists, output)

    for singleModel in netModel.modules():
        singleModelSlices = list(singleModel.modules())
        if len(singleModelSlices) == 1:
            hnd = singleModel.register_forward_hook(moduleParams_hook)
            print("$$$$$$$$$$$$$$$$$$$ Layer hooking$$$$$$$$$$$$$ ",inum)
            inum = inum + 1
            global_hooks_handles.append(hnd)
    ###here could hook funcs
    funcsParams_hook()

    return None
def hookwork_assembed():
    blobSize = 0
    if len(TorchLayers_Keeper) == 0:
        print("Make sure interpreter is working...")
    for item in TorchLayers_Keeper[:]:
        if len(item['inputs_layersName']) == 0:
            TorchLayers_Keeper.remove(item)
    ####################add outputs key#################
    lysize = len(TorchLayers_Keeper)
    ###################draw graph here is ok##############################
    if lysize > 0:
        g_pen = TorchGraph_Pen(TorchLayers_Keeper, "./outputs/ncnn")
        g_pen.makeLayer()
    for ll in range(0, lysize):
        cur_layer = TorchLayers_Keeper[ll]
        cur_layer_name = cur_layer['cur_layer_Name']
        cur_layer['outputs_layersName'] = []
        cur_layer_outputsname_list = []
        indx = ll
        for next_layer in TorchLayers_Keeper[(ll+1):lysize]:    #last - 1
            indx += 1
            next_layer_inputs = next_layer['inputs_layersName']
            if cur_layer_name in next_layer_inputs:
                cur_layer['splitCnt'] += 1
                blobSize += 1
                cur_layer_outputsname_list.append(indx)

        if cur_layer['splitCnt'] < 2:
            cur_layer['outputs_layersName'].append(cur_layer_name)
            if cur_layer['splitCnt'] == 0:
                blobSize = blobSize + 1
        else:
            for k in range(0,cur_layer['splitCnt']):
                snm = cur_layer_name + '_splitncnn_' + str(k)
                cur_layer['outputs_layersName'].append(snm)
                lidx = cur_layer_outputsname_list[k]
                outlyr = TorchLayers_Keeper[lidx]
                if len(outlyr['inputs_layersName']) == 1:
                    outlyr['inputs_layersName'][0] = snm
                else:
                    for ik in range(0,len(outlyr['inputs_layersName'])):
                        if outlyr['inputs_layersName'][ik] == cur_layer_name:
                            outlyr['inputs_layersName'][ik] = snm

    print("layer interpreter is ok now.....")
    return blobSize

def unMount_forward_hooks():
    for hookhnd in global_hooks_handles:
        hookhnd.remove()
    return None

def convert_walking(dstType = 0):
    blobNum = hookwork_assembed() ###call once###
    if dstType == FRAME_WORK_TYPE["DEFAULT"]:
        NCNN_NET = NCNN_FRAMEWORK_FACTORY("./outputs/")
        NCNN_NET.converting(TorchLayers_Keeper, blobNum)
    elif dstType == FRAME_WORK_TYPE["CAFFE"]:
    #elif FRAME_WORK_TYPE == FRAME_WORK_TYPE["TENSOR_FLOW"]:
    #elif FRAME_WORK_TYPE == FRAME_WORK_TYPE["ONNX"]:
        print("caffe is now")
    else:
        print("Sorry, some AI framework can not be supported...")


def graph_walking():
    print("draw network flow-graphic")



