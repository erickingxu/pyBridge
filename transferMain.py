########################################################################################################################
##############This is your forward network main file for loading trained model and show reslut in these codes###########
########################################################################################################################
from torch.autograd import Variable, Function  ##must import for using pytorch hooks function
import hookWorker as HookWorker

old_function__call__ = Function.__call__
global_hooks_handles = []
#########################################Hooks logical##################################################################
def params_hook(NetModule, input, output):
    inLists = list(input) # listed for tensor for hooks
    HookWorker.forward_hooking(NetModule, inLists, output)

def forward_hooks_mount2_Net(netModel):
    for singleModel in netModel.modules():
        singleModelSlices = list(singleModel.modules())
        if len(singleModelSlices) == 1:
            hnd = singleModel.register_forward_hook(params_hook)
            global_hooks_handles.append(hnd)
    return None

def unMount_forward_hooks():
    for hookhnd in global_hooks_handles:
        hookhnd.remove()
    return None
##########################################Main logical###################################################################
def main:
##First step find your checkpoints model loader place and then hung on my hook into your every slice model##
##load model from ck or in other way ,like load_state_dict method
##support cpu mode or gpu ParallelData mode for pytorch(could escape from ONNX bugs)##
    NetModel = 'your model'
    inputShapeData = torch.rand(1, 3, 256, 256) #just for net-foward once is enough
    # step 1: register hooks for module or functions
    forward_hooks_mount2_Net(NetModel)
    # step 2: forward the net work
    output = NetModel(inputShapeData)
    # step 3: remove the  hooks
    unMount_forward_hooks()
    # step 4: interpret into NCNN layer's weight and bias in ncnn param-bin file format


if __name__ == '__main__':
    args = 'get model args whatever'
    main(args)