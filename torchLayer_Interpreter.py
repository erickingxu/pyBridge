import  numpy as py
import torch
import torch.nn as nn
from torchLayer_IMPL import PYTORCH_MODULE_SUPPORTER, PYTORCH_FUNCTION_SUPPORTER
import pdb
##################################################################

###############################################################################################
###############################################################################################
#################################CORE CLASS FOR INTERPRETER####################################
###############################################################################################
###############################################################################################
once_func_hooked = True
TorchLayers_Keeper = []
#TorchOps_Keeper = {}
class  Layer_Interpreter(object):
    def __init__(self, moduleFuncs, inLists, output):
        self.inputs = inLists
        self.module_funcs  = moduleFuncs
        self.output = output
        #self.once_func_hooked = True
        self.torchOP_Dicts = {}
        global TorchLayers_Keeper
        if TorchLayers_Keeper is None:
            TorchLayers_Keeper = []

    def __str__(self):
        return str(self.torchOP_Dicts)

    @staticmethod
    def funcs_Interpretering():
        for funcKey in PYTORCH_FUNCTION_SUPPORTER.keys():
            print("......Pytorch function is hooked now for....", funcKey)
            PYTORCH_FUNCTION_SUPPORTER[funcKey](TorchLayers_Keeper)  ########neeeed to be store in layer....
        #TorchLayers_Keeper.append(TorchOps_Keeper)
        return True

    def Layers_Interpretering(self, *args, **kwargs):
        ret = False
        support_key = self.module_funcs.__class__.__name__

        if isinstance(self.module_funcs, nn.Module):
            if support_key in PYTORCH_MODULE_SUPPORTER.keys():
                ret = True
                #print(".....Pytorch module is interpreting for....", self.module_funcs)
                self.torchOP_Dicts = PYTORCH_MODULE_SUPPORTER[support_key](self.module_funcs, self.inputs, self.output)
                if len(self.torchOP_Dicts) == 0:
                    ret = False
                    return ret

                TorchLayers_Keeper.append(self.torchOP_Dicts)
        else:
            print("xxxxxxxxxxxxPytorch OP Can not interpreting xxxxxxxxxxxxx",support_key)
            ret = False
            pdb.set_trace()
            assert (False)
        ###########################################################################
        #########################WORK ONCE#########################################
        #self.func_hooked_loop()
        ###########################################################################
        return ret
