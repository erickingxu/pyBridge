import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import hashlib

###############################################################################################
###############################################################################################
###################Supporter OF Modules and Funcs FOR INTERPRETER##############################
###############################################################################################
###############################################################################################

TYPE_ACTIVATION_FUNC = {
    'None'      :            0,
    'Linear'    :            1,
    'Relu'      :            2,
    'TanH'      :            3,
    'Sigmoid'   :            4,
    'Absolute'  :            5,
    'Leaky'     :            6,

}

TYPE_ELEMWISE_OP = {
    'Add':                   0,
    'Mul':                   1,
    'Div':                   2,
    'Max':                   3,
    'Cat':                   4,
    'Relu':                  5,
    'Reshape':               6,
    'Permute':               7,
    'SoftMax':               8,
    'ReduceMean':            9,
}

# TYPE_TENSOR_OP = {
#     'Permute':              0,
#     'Sub':                  1,
# }

TYPE_POOLING = {
    'Max'       :            0,
    'Avg'       :            1,
}

TYPE_UPSAMPLE = {
    'Nearest':  0,
    'BiLinear':  1,
    'Bicubic':  2
}

def create_StrUID(inStr):
    strMd5 = hashlib.md5(inStr.encode("utf-8"))
    strID  = strMd5.hexdigest()
    return str(strID)

gl_type_dicts_count = {}
gl_layer_lists = []
gl_layer_NUM = 0
def module_Interpreter_For_Conv2D(module, inputs, output):
    #####################################################
    inp_id = id(inputs[0])
    outp_id = id(output)
    if inp_id == outp_id:
        return {}
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Convolution'] = 1
    else:
        tn = 'Convolution'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Convolution'] += 1
        else:
            gl_type_dicts_count['Convolution'] = 1
    convGroup = module.groups
    cvtypename = "Convolution"
    if convGroup > 1:
        cvtypename = "ConvolutionDepthWise"

    layer_name = cvtypename + str(gl_type_dicts_count['Convolution'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = [] #[layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []
    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break
    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################
    torchOP_LayerInfos                                      = {}
    torchOP_LayerInfos['cur_layer_Name']                    = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Convolution'
    torchOP_LayerInfos['inputs_layersName']                 = layer_inputs

    torchOP_LayerInfos['number_of_Input_Feature_Channels']  = inputs[0].shape[1]
    torchOP_LayerInfos['height_of_Input_Feature']           = inputs[0].shape[2]
    torchOP_LayerInfos['width_of_Input_Feature']            = inputs[0].shape[3]
    torchOP_LayerInfos['number_of_Output_Feature_Channels'] = output.shape[1]
    torchOP_LayerInfos['height_of_Output_Feature']          = output.shape[2]
    torchOP_LayerInfos['width_of_Output_Feature']           = output.shape[3]
    torchOP_LayerInfos['kernel_Height']                     = module.kernel_size[0]
    torchOP_LayerInfos['kernel_Width']                      = module.kernel_size[1]
    torchOP_LayerInfos['padding_Size_Y']                    = module.padding[0]
    torchOP_LayerInfos['padding_Size_X']                    = module.padding[1]
    torchOP_LayerInfos['stride_Size_Y']                     = module.stride[0]
    torchOP_LayerInfos['stride_Size_X']                     = module.stride[1]
    torchOP_LayerInfos['dilation_Size_Y']                   = module.dilation[0]
    torchOP_LayerInfos['dilation_Size_X']                   = module.dilation[1]
    torchOP_LayerInfos['activation_Type']                   = TYPE_ACTIVATION_FUNC['None']
    torchOP_LayerInfos['num_of_Group']                      = module.groups
    torchOP_LayerInfos['splitCnt'] = 0
    if module.bias is None:
        torchOP_LayerInfos['if_Use_Bias']                   = 0
        # cvvv = module.weight.data
        # cwww = cvvv.numpy()
        # ctt = torch.rand(4)
        # ctw0 = ctt.numpy().astype(np.float32)
        # ctw = np.around(ctt.numpy(),decimals=5).astype(np.float32)
        covnweights = module.weight.detach().numpy()
        torchOP_LayerInfos['weights_Numpy_NCHW_F32']        = covnweights.astype('float32') #np.around(module.weight.detach().numpy().ravel(),decimals=4)
        torchOP_LayerInfos['bias_Numpy_F32'] = np.zeros((output.shape[1],), dtype=np.float32).astype(np.float32)
    else:
        torchOP_LayerInfos['if_Use_Bias']                   = 1
        convwt = module.weight.detach().numpy()
        torchOP_LayerInfos['weights_Numpy_NCHW_F32']        = convwt.astype(np.float32)
        convbias = module.bias.detach().numpy()
        torchOP_LayerInfos['bias_Numpy_F32'] = convbias.astype(np.float32)

    return torchOP_LayerInfos


def module_Interpreter_For_BatchNorm(module, inputs, output):
    #####################################################
    inp_id = id(inputs[0])
    outp_id = id(output)
    if inp_id == outp_id:
        return {}

    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['BatchNorm'] = 1
    else:
        tn = 'BatchNorm'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['BatchNorm'] += 1
        else:
            gl_type_dicts_count['BatchNorm'] = 1

    layer_name = "BatchNorm" + str(gl_type_dicts_count['BatchNorm'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break

    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################
    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'BatchNorm'

    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['number_of_InOut_Feature_Channels'] = output.shape[1]
    torchOP_LayerInfos['height_of_InOut_Feature'] = output.shape[2]
    torchOP_LayerInfos['width_of_InOut_Feature'] = output.shape[3]
    npweight = module.weight.detach().numpy()
    bnbias = module.bias.detach().numpy()
    bnvar = module.running_var.detach().numpy()
    bnmean = module.running_mean.detach().numpy()
    torchOP_LayerInfos['slope_data'] = npweight.astype(np.float32)
    torchOP_LayerInfos['bias_data'] = bnbias.astype(np.float32)
    torchOP_LayerInfos['var_data'] = bnvar.astype(np.float32)
    torchOP_LayerInfos['mean_data'] = bnmean.astype(np.float32)
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos

def module_Interpreter_For_Relu(module, inputs, output):
    # if output.equal(inputs[0]):
    #     return {}
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['ReLU'] = 1
    else:
        tn = 'ReLU'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['ReLU'] += 1
        else:
            gl_type_dicts_count['ReLU'] = 1

    layer_name = "ReLU" + str(gl_type_dicts_count['ReLU'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []
    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])

        if len(layer_inputs) == 1:
            break

    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################

    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'ReLU'

    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['number_of_InOut_Feature_Channels'] = output.shape[1]
    torchOP_LayerInfos['height_of_InOut_Feature'] = output.shape[2]
    torchOP_LayerInfos['width_of_InOut_Feature'] = output.shape[3]
    torchOP_LayerInfos['activation_Type'] = TYPE_ACTIVATION_FUNC['Relu']
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos
################################################################################
def module_Interpreter_For_MLinear(module, inputs, output):
    #####################################################
    inp_id = id(inputs[0])
    outp_id = id(output)
    if inp_id == outp_id:
        return {}

    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['MLinear'] = 1
    else:
        tn = 'MLinear'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['MLinear'] += 1
        else:
            gl_type_dicts_count['MLinear'] = 1

    layer_name = "MLinear" + str(gl_type_dicts_count['MLinear'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break

    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################
    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'MLinear'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    mlweight = module.weight.detach().numpy()
    mlbias = module.bias.detach().numpy()
    torchOP_LayerInfos['weights_data'] = mlweight.astype(np.float32)
    torchOP_LayerInfos['bias_data'] = mlbias.astype(np.float32)

    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos
################################################


def getUpsampleFunctionArguments(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
        return {"size": size, "scale_factor": scale_factor, "mode": mode, "align_corners": align_corners }


def getPoolingFunctionArguments(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
                                return_indices=False):
    return {"kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode,
            "dilation": dilation, "return_indices": return_indices}

def getPermuteFunctionArguments(input,*dims):
    return {"dims":dims}

def getReshapeFunctionArguments(input,*shape):
    return {"shape":shape}
def getSoftMaxFunctionArguments(input,dim):
    return {"dim":dim}
dict_origin_F = {}


def func_forward_hooking(func, output, func_infos_List=[], *args, **kwargs):
    inputs = [a for a in args if isinstance(a, Variable)]
    inputs += [a for a in kwargs.values() if isinstance(a, Variable)]
    if len(inputs) == 0:
        inputs = [a for a in args[0] if isinstance(a, Variable)]
        inputs += [a for a in kwargs.values() if isinstance(a, Variable)]
    if func in AVALIABLE_FUNCTION_CONVERTER.keys():
        func_ops_Info = {}
        func_ops_Info = AVALIABLE_FUNCTION_CONVERTER[func](func, inputs, output, *args, **kwargs)

        if len(func_ops_Info) == 0:
            return False

        func_infos_List.append(func_ops_Info)
        #print("@@@@@@@converting function@@@@@@@@ ", func_infos_dict)
    else:
        print("unsupport function converter of ",func)

    return True

def overload_pooling(funcs_dict=[]):
    dict_origin_F['OVER_LOOAD_POOLING'] = {}
    dict_origin_F['OVER_LOOAD_POOLING'][TYPE_POOLING['Avg']] = F.avg_pool2d
    def new_functional_avg_pool2d(*args, **kwargs):
        output = dict_origin_F['OVER_LOOAD_POOLING'][TYPE_POOLING['Avg']](*args, **kwargs)
        # add hook func
        func_forward_hooking(dict_origin_F['OVER_LOOAD_POOLING'][TYPE_POOLING['Avg']], output, funcs_dict,*args, **kwargs)

        return output
    F.avg_pool2d = new_functional_avg_pool2d
    #
    # next overload avg_pool2d
    dict_origin_F['OVER_LOOAD_POOLING'][TYPE_POOLING['Max']] = F.max_pool2d
    def new_functional_max_pool2d(*args, **kwargs):
        output = dict_origin_F['OVER_LOOAD_POOLING'][TYPE_POOLING['Max']](*args, **kwargs)
        # add hook func
        func_forward_hooking(dict_origin_F['OVER_LOOAD_POOLING'][TYPE_POOLING['Max']], output, funcs_dict,*args, **kwargs)

        return output
    F.max_pool2d = new_functional_max_pool2d
    return None


def overload_upsampling(funcs_dict=[]):
    dict_origin_F['OVER_LOOAD_UPSAMPLING'] = F.interpolate

    def new_functional_upsample(*args, **kwargs):
        output = dict_origin_F['OVER_LOOAD_UPSAMPLING'](*args, **kwargs)
        # add hook func
        func_forward_hooking(dict_origin_F['OVER_LOOAD_UPSAMPLING'], output,funcs_dict, *args, **kwargs)

        return output
    F.interpolate = new_functional_upsample
    return None

def overload_elementwiseoperator(funcs_dict=[]):
    dict_origin_F['ElementWiseOperator'] = {}
    #
    # next overload add
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Add']] = torch.add

    def new_torch_add(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Add']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Add']],output,funcs_dict, *args, **kwargs)

        return output

    torch.add = new_torch_add
    #
    # next overload mul
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Mul']] = torch.mul

    def new_torch_mul(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Mul']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Mul']],output,funcs_dict, *args, **kwargs)

        return output

    torch.mul = new_torch_mul
    #
    # next overload div
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Div']] = torch.div

    def new_torch_div(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Div']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Div']],output, funcs_dict, *args, **kwargs)

        return output

    torch.div = new_torch_div
    #
    # next overload max
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Max']] = torch.max

    def new_torch_max(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Max']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Max']],output, funcs_dict, *args, **kwargs)

        return output

    torch.max = new_torch_max

    # next overload softmax
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['SoftMax']] = F.softmax
    def new_softmax(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['SoftMax']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['SoftMax']], output, funcs_dict, *args,
                             **kwargs)

        return output

    F.softmax = new_softmax
    #
    # next overload cat
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Cat']] = torch.cat
    def new_torch_cat(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Cat']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Cat']],output,funcs_dict, *args, **kwargs)

        return output

    torch.cat = new_torch_cat
    #
    # next overload relue
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Relu']] = F.relu

    def new_torch_relu(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Relu']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Relu']], output, funcs_dict, *args,
                             **kwargs)

        return output
    F.relu = new_torch_relu
    # next overload permute
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Permute']] = torch.Tensor.permute
    def new_torch_permute(*args, **kwargs):
        #dims = (args[1], args[2],args[3],args[4])
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Permute']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Permute']], output, funcs_dict, *args,
                             **kwargs)

        return output
    torch.Tensor.permute = new_torch_permute

    # next overload reshape
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Reshape']] = torch.Tensor.reshape
    def new_torch_reshape(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Reshape']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['Reshape']], output, funcs_dict,
                             *args,
                             **kwargs)

        return output
    torch.Tensor.reshape = new_torch_reshape

    # next overload reduceMean
    dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['ReduceMean']] = torch.mean

    def new_torch_reducemean(*args, **kwargs):
        output = dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['ReduceMean']](*args, **kwargs)
        func_forward_hooking(dict_origin_F['ElementWiseOperator'][TYPE_ELEMWISE_OP['ReduceMean']], output, funcs_dict,
                             *args,
                             **kwargs)

        return output

    torch.mean = new_torch_reducemean
    return None
##################
################################################################################################
################################################################################################
def function_Interpreter_For_ElementWiseOp_Add(func, inputs, output, *args, **kwargs):
    inp_ids = [id(inp) for inp in inputs]
    outp_id = id(output)
    if outp_id in inp_ids:
        return {}
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Add'] = 1
    else:
        tn = 'Add'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Add'] += 1
        else:
            gl_type_dicts_count['Add'] = 1

    layer_name = "Add" + str(gl_type_dicts_count['Add'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs)
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 2:
            break

    if len(layer_inputs) < len(inputs):  # must <=
        if 1 == (len(inputs) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert (False)
            lyn = gl_layer_lists[k - 1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs):  # some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################
    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Add'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['number_of_Input_Layers'] = len(inputs)
    torchOP_LayerInfos['operator_Type'] = TYPE_ELEMWISE_OP['Add']
    torchOP_LayerInfos['activation_Type'] = TYPE_ACTIVATION_FUNC['None']
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos

def function_Interpreter_For_ElementWiseOp_ArrMax(func, inputs, output, *args, **kwargs):

    inp_ids = [id(inp) for inp in inputs[0]]
    outp_id = id(output)
    if outp_id in inp_ids:
        return {}

    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Max'] = 1
    else:
        tn = 'Max'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Max'] += 1
        else:
            gl_type_dicts_count['Max'] = 1

    layer_name = "Max" + str(gl_type_dicts_count['Max'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break

    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert (False)

            lyn = gl_layer_lists[k - 1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):  # some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################
    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Max'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['operator_Type']     = TYPE_ELEMWISE_OP['Max']
    torchOP_LayerInfos['activation_Type']   = TYPE_ACTIVATION_FUNC['None']
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos


def function_Interpreter_For_ElementWiseOp_Cat(func, inputs, output, *args, **kwargs):
    if len(inputs) == 1:
        ts_input = inputs[0]
        ts_output = output
        if ts_input.equal(ts_output):
            return {}
        else:
            print("Concat change sth")

    inp_ids = [inp for inp in inputs]
    outp_id = id(output)
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Concat'] = 1
    else:
        tn = 'Concat'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Concat'] += 1
        else:
            gl_type_dicts_count['Concat'] = 1

    layer_name = "Concat" + str(gl_type_dicts_count['Concat'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    inp_list = []
    for i_inp in inputs:
        inp_list.append(i_inp)

    l_list.append(inp_list)
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []
    inps_dict = {}
    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        inps_i = 0
        for i_inp in inp_list:
            inps_i += 1
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
                    inps_dict[inps_i - 1] = gl_layer_lists[curIndex - 1][0]

        if len(layer_inputs) == len(inp_list):
            if len(inps_dict) > 1:
                for idxx in range(0, len(inps_dict)):
                    layer_inputs[idxx] = inps_dict[idxx]
            pass
            break

    if len(layer_inputs) < len(inputs):  # must <=
        if 1 == (len(inputs) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert (False)

            lyn = gl_layer_lists[k - 1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################

    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name']            = layer_name
    torchOP_LayerInfos["layer_Type"]                = "Concat"

    torchOP_LayerInfos['inputs_layersName']         = layer_inputs

    torchOP_LayerInfos['number_of_Input_Layers']    = len(inputs)

    torchOP_LayerInfos['number_of_InOut_Feature_Channels'] = output.shape[0]
    torchOP_LayerInfos['Concat_Dim']   = args[1]
    torchOP_LayerInfos['operator_Type']             = TYPE_ELEMWISE_OP['Cat']
    torchOP_LayerInfos['activation_Type']           = TYPE_ACTIVATION_FUNC['None']
    torchOP_LayerInfos['splitCnt']                  = 0

    return torchOP_LayerInfos
#####################################################################################
def function_Interpreter_For_Relu(func,inputs, output,*args, **kwargs):
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['ReLU'] = 1
    else:
        tn = 'ReLU'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['ReLU'] += 1
        else:
            gl_type_dicts_count['ReLU'] = 1

    layer_name = "ReLU" + str(gl_type_dicts_count['ReLU'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []
    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])

        if len(layer_inputs) == 1:
            break

    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert (False)

            lyn = gl_layer_lists[k - 1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):  # some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################

    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'ReLU'

    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['number_of_InOut_Feature_Channels'] = output.shape[1]
    torchOP_LayerInfos['height_of_InOut_Feature'] = output.shape[2]
    torchOP_LayerInfos['width_of_InOut_Feature'] = output.shape[3]
    torchOP_LayerInfos['activation_Type'] = TYPE_ACTIVATION_FUNC['Relu']
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos
#####################################################################################
def function_Interpreter_For_ReduceMean(func, inputs, output,*args, **kwargs):
    if output.equal(inputs[0]):
        print("..... now is permute no work......")
        return {}
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['ReduceMean'] = 1
    else:
        tn = 'ReduceMean'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['ReduceMean'] += 1
        else:
            gl_type_dicts_count['ReduceMean'] = 1

    layer_name = "ReduceMean" + str(gl_type_dicts_count['ReduceMean'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break
    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert (False)

            lyn = gl_layer_lists[k - 1][0]
            layer_inputs.append(lyn)
        else:
            print("Permute....fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):  # some output is same ,please take nearest one
        print("Permute....cather wrong layer.....")
    ######################################################

    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'ReduceMean'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['params'] = (3 ,2 ,0.0625)
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos
#####################################################################################
def function_Interpreter_For_Permute(func, inputs, output,*args, **kwargs):
    if output.equal(inputs[0]):
        print("..... now is permute no work......")
        return {}
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Permute'] = 1
    else:
        tn = 'Permute'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Permute'] += 1
        else:
            gl_type_dicts_count['Permute'] = 1

    layer_name = "Permute" + str(gl_type_dicts_count['Permute'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break
    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert (False)

            lyn = gl_layer_lists[k - 1][0]
            layer_inputs.append(lyn)
        else:
            print("Permute....fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):  # some output is same ,please take nearest one
        print("Permute....cather wrong layer.....")
    ######################################################
    args_dict = getPermuteFunctionArguments(*args, **kwargs)
    torchOP_LayerInfos = {}
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Permute'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['dims'] = args_dict['dims']
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos

#######################################################################################
def function_Interpreter_For_Reshape(func, inputs, output, *args, **kwargs):
    if output.equal(inputs[0]):
        print("..... now is reshape ......")
        return {}
#####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Reshape'] = 1
    else:
        tn = 'Reshape'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Reshape'] += 1
        else:
            gl_type_dicts_count['Reshape'] = 1

    layer_name = "Reshape" + str(gl_type_dicts_count['Reshape'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break
    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("Reshape....fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("Reshape....cather wrong layer.....")
    ######################################################

    torchOP_LayerInfos = {}
    args_dict = getReshapeFunctionArguments(*args, **kwargs)
    rsparms = args_dict['shape']
    tupleNum = len(rsparms)
    nsz = 1
    tsize = inputs[0].size()
    for i in range(len(tsize)):
        nsz = nsz * inputs[0].size(i)
    reshap_params = (rsparms[-1], round(nsz / rsparms[-1]), 1)

    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Reshape'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['shap'] = reshap_params
    torchOP_LayerInfos['splitCnt'] = 0
    return torchOP_LayerInfos
#######################################################################################
def function_Interpreter_For_Pooling(func, inputs, output, *args, **kwargs):

    if output.equal(inputs[0]):
        return {}
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Pooling'] = 1
    else:
        tn = 'Pooling'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Pooling'] += 1
        else:
            gl_type_dicts_count['Pooling'] = 1

    layer_name = "Pooling" + str(gl_type_dicts_count['Pooling'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break
    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################

    torchOP_LayerInfos = {}
    args_dict = getPoolingFunctionArguments(*args, **kwargs)
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Pooling'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs

    torchOP_LayerInfos['number_of_Input_Feature_Channels'] = inputs[0].shape[1]
    torchOP_LayerInfos['height_of_Input_Feature'] = inputs[0].shape[2]
    torchOP_LayerInfos['width_of_Input_Feature'] = inputs[0].shape[3]
    torchOP_LayerInfos['number_of_Output_Feature_Channels'] = output.shape[1]
    torchOP_LayerInfos['height_of_Output_Feature'] = output.shape[2]
    torchOP_LayerInfos['width_of_Output_Feature'] = output.shape[3]
    torchOP_LayerInfos['splitCnt'] = 0
    # get kernel size:
    if isinstance(args_dict['kernel_size'], list) or isinstance(args_dict['kernel_size'], tuple):
        kernel_sizes = list()
        if len(kernel_sizes) == 1:
            torchOP_LayerInfos['kernel_Height'] = kernel_sizes[0]
            torchOP_LayerInfos['kernel_Width'] = kernel_sizes[0]
        elif len(kernel_sizes) == 2:
            torchOP_LayerInfos['kernel_Height'] = kernel_sizes[0]
            torchOP_LayerInfos['kernel_Width'] = kernel_sizes[1]
        else:
            assert (False)
    else:
        torchOP_LayerInfos['kernel_Height'] = args_dict['kernel_size']
        torchOP_LayerInfos['kernel_Width'] = args_dict['kernel_size']

    # get padding size:
    if isinstance(args_dict['padding'], list) or isinstance(args_dict['padding'], tuple):
        padding_sizes = list(args_dict['padding'])
        if len(padding_sizes) == 1:
            torchOP_LayerInfos['padding_Size_Y'] = padding_sizes[0]
            torchOP_LayerInfos['padding_Size_X'] = padding_sizes[0]
        elif len(padding_sizes) == 2:
            torchOP_LayerInfos['padding_Size_Y'] = padding_sizes[0]
            torchOP_LayerInfos['padding_Size_X'] = padding_sizes[1]
        else:
            assert (False)
    else:
        torchOP_LayerInfos['padding_Size_Y'] = args_dict['padding']
        torchOP_LayerInfos['padding_Size_X'] = args_dict['padding']

    # get stride size:
    # import pdb; pdb.set_trace()
    if args_dict['stride'] is None:
        args_dict['stride'] = args_dict['kernel_size']
    if isinstance(args_dict['stride'], list) or isinstance(args_dict['stride'], tuple):
        stride_sizes = list(args_dict['stride'])
        if len(padding_sizes) == 1:
            torchOP_LayerInfos['stride_Size_Y'] = stride_sizes[0]
            torchOP_LayerInfos['stride_Size_X'] = stride_sizes[0]
        elif len(padding_sizes) == 2:
            torchOP_LayerInfos['stride_Size_Y'] = stride_sizes[0]
            torchOP_LayerInfos['stride_Size_X'] = stride_sizes[1]
        else:
            assert (False)
    else:
        torchOP_LayerInfos['stride_Size_Y'] = args_dict['stride']
        torchOP_LayerInfos['stride_Size_X'] = args_dict['stride']

    # get pooling type:
    if func == dict_origin_F['OVER_LOOAD_POOLING'][
        TYPE_POOLING['Max']]:
        torchOP_LayerInfos['pooling_Type'] = TYPE_POOLING['Max']
    elif func == dict_origin_F['OVER_LOOAD_POOLING'][TYPE_POOLING['Avg']]:
        torchOP_LayerInfos['pooling_Type'] = TYPE_POOLING['Avg']
    else:
        print("unsupported Pooling Type Functons.")
        assert (False)
    return torchOP_LayerInfos


def function_Interpreter_For_Upsampling(func, inputs, output, *args, **kwargs):
    if output.equal(inputs[0]):
        return {}
    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Upsample'] = 1
    else:
        tn = 'Upsample'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Upsample'] += 1
        else:
            gl_type_dicts_count['Upsample'] = 1

    layer_name = "Upsample" + str(gl_type_dicts_count['Upsample'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break
    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs[0]) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################
    torchOP_LayerInfos = {}
    args_dict = getUpsampleFunctionArguments(*args, **kwargs)
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Upsample'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs

    torchOP_LayerInfos['number_of_InOut_Feature_Channels'] = output.shape[1]
    torchOP_LayerInfos['height_of_Input_Feature'] = inputs[0].shape[2]
    torchOP_LayerInfos['width_of_Input_Feature'] = inputs[0].shape[3]
    torchOP_LayerInfos['height_of_Output_Feature'] = output.shape[2]
    torchOP_LayerInfos['width_of_Output_Feature'] = output.shape[3]
    torchOP_LayerInfos['upscale_Factor'] = args_dict['scale_factor']
    torchOP_LayerInfos['upsample_AlignCorner'] = 0
    torchOP_LayerInfos['splitCnt'] = 0
    if args_dict['align_corners'] is not None:
        if args_dict['align_corners']:
            torchOP_LayerInfos['upsample_AlignCorner'] = 1
    if args_dict['mode'] == 'nearest':
        torchOP_LayerInfos['upsample_Type'] = TYPE_UPSAMPLE['Nearest']
    elif args_dict['mode'] == 'bilinear':
        torchOP_LayerInfos['upsample_Type'] = TYPE_UPSAMPLE['BiLinear']
    elif args_dict['mode'] == 'bicubic':
        torchOP_LayerInfos['upsample_Type'] = TYPE_UPSAMPLE['Bicubic']
    else:
        print("unsupport upsample type ", args_dict['mode'])
        assert (False)
    return torchOP_LayerInfos

def function_Interpreter_For_SoftMax(func, inputs, output, *args, **kwargs):

    #####################################################
    global gl_type_dicts_count
    global gl_layer_lists
    global gl_layer_NUM
    layer_name = ""
    if len(gl_type_dicts_count) == 0:
        gl_type_dicts_count['Softmax'] = 1
    else:
        tn = 'Softmax'
        if tn in gl_type_dicts_count.keys():
            gl_type_dicts_count['Softmax'] += 1
        else:
            gl_type_dicts_count['Softmax'] = 1

    layer_name = "Softmax" + str(gl_type_dicts_count['Softmax'] - 1)
    gl_layer_NUM += 1
    #####################################################
    l_list = []  # [layer_name,inputs,output]
    l_list.append(layer_name)
    l_list.append(inputs[0])
    l_list.append(output)
    if len(gl_layer_lists) > gl_layer_NUM:
        gl_layer_lists[gl_layer_NUM - 1] = l_list
    else:
        gl_layer_lists.append(l_list)

    layer_inputs = []

    for j in range(0, gl_layer_NUM):
        curIndex = gl_layer_NUM - 1 - j
        for i_inp in inputs:
            if curIndex == 0:
                if i_inp.equal(gl_layer_lists[0][1]):  # first
                    layer_inputs.append("0")
            else:
                if i_inp.equal(gl_layer_lists[curIndex - 1][2]):  # first
                    layer_inputs.append(gl_layer_lists[curIndex - 1][0])
        if len(layer_inputs) == 1:
            break
    if len(layer_inputs) < len(inputs[0]):  # must <=
        if 1 == (len(inputs) - len(layer_inputs)):  ###Attention!!! maybe some data changed bettween layers
            # force no-owned input from last layer
            k = len(gl_layer_lists) - 1
            if k <= 0:
                assert(False)

            lyn = gl_layer_lists[k-1][0]
            layer_inputs.append(lyn)
        else:
            print("fuck error crashed....")
    elif len(layer_inputs) > len(inputs[0]):#some output is same ,please take nearest one
        print("cather wrong layer.....")
    ######################################################
    torchOP_LayerInfos = {}
    args_dict = getSoftMaxFunctionArguments(*args, **kwargs)
    torchOP_LayerInfos['cur_layer_Name'] = layer_name
    torchOP_LayerInfos['layer_Type'] = 'Softmax'
    torchOP_LayerInfos['inputs_layersName'] = layer_inputs
    torchOP_LayerInfos['softmax_Dim'] = args_dict['dim']
    torchOP_LayerInfos['splitCnt'] = 0

    return torchOP_LayerInfos
################################################################################################
#################################################################################################
PYTORCH_MODULE_SUPPORTER = {
###KEY### : ###VALUE_FOR_IMPL###
    nn.modules.conv.Conv2d.__name__:              module_Interpreter_For_Conv2D,
    nn.modules.batchnorm.BatchNorm2d.__name__:    module_Interpreter_For_BatchNorm,
    nn.modules.activation.ReLU.__name__:          module_Interpreter_For_Relu,
    nn.modules.Linear.__name__:                   module_Interpreter_For_MLinear,

#    nn.modules.pooling.AvgPool2d.__name__:        module_Interpreter_For_AvgPool2d,
#    nn.modules.upsampling.Upsample.__name__:      function_Interpreter_For_Upsampling,
}

PYTORCH_FUNCTION_SUPPORTER = {
    'Pooling'                   :   overload_pooling,
    'Upsampling'                :   overload_upsampling,
    'ElementWiseOperator'       :   overload_elementwiseoperator,
}
AVALIABLE_FUNCTION_CONVERTER = {
###KEY### : ###VALUE_FOR_IMPL###
    torch.add       :                                 function_Interpreter_For_ElementWiseOp_Add,
    torch.max       :                                 function_Interpreter_For_ElementWiseOp_ArrMax,
    torch.cat       :                                 function_Interpreter_For_ElementWiseOp_Cat,
    torch.mean      :                                 function_Interpreter_For_ReduceMean,
    F.relu          :                                 function_Interpreter_For_Relu,
    torch.Tensor.permute   :                          function_Interpreter_For_Permute,
    torch.Tensor.reshape   :                          function_Interpreter_For_Reshape,
    F.avg_pool2d    :                                 function_Interpreter_For_Pooling,
    F.max_pool2d    :                                 function_Interpreter_For_Pooling,
    F.interpolate   :                                 function_Interpreter_For_Upsampling,
    F.softmax       :                                 function_Interpreter_For_SoftMax,
}
