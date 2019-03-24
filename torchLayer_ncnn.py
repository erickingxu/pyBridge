
import os
import fileinput

try:
    import numpy as np
except:
    os.system('conda install numpy')
    import numpy as np

class  NCNN_FRAMEWORK_FACTORY(object):
    def  __init__(self, outDir = ''):
        self.paramFilePath      = outDir + 'ncnn.param'
        self.binFilePath        = outDir + 'ncnn.bin'
        self.layer_count    = 0
        self.blob_count     = 0
        self.net            = {}
        self.layerSequential = []
        if os.path.exists(self.binFilePath):
            os.remove(self.binFilePath)

    def __str__(self):
        return str(self.net)

    #0 => float32, 0x01306B47 => float16, otherwise => quantized int8
    def headWriter(self, path = None):
        if path is None:
            assert "nO header is no ncnn...."
        pass
        with open(path, "ab+") as fw:
            fw.write(np.uint32(0).tobytes())

    def reWrite_locLine(self, filename, lineNo, newText, bLeft=True, bRight=False):
        if os.path.exists(filename):
            f = fileinput.input(filename, inplace=1)
            for line in f:
                line_txt = line.replace("\r\n","").replace("\n","")
                if f.lineno() == lineNo:
                    if bLeft:
                        print(line_txt + newText)
                    elif bRight:
                        print(newText + line_txt)
                    else:
                        print(newText)
                else:
                    print(line_txt)
        else:
            print("No file can be rewrotten now....")

    def converting(self, torchLayer_List, blob_num):
        layersNum = len(torchLayer_List)
        if layersNum == 0:
            print("make sure...layer interpretor is running before...")
        ncnn_layer_Num = 0
        ncnn_blob_Num  = 0
#####################save input layer into param file##################
        with open(self.paramFilePath, "w") as fparam:
            strr = 7767517
            fparam.write('%d\n' % (strr))
            fparam.write('%d %d\n' % (layersNum, blob_num)) #num is not ncnn's, not include split layer
            fparam.write('%s            %s            %d %d     %d\n' % ('Input', '0', 0,1,0))
            ncnn_layer_Num += 1
            ncnn_blob_Num += 1
            for layer_Info in torchLayer_List:
                inp_layersName = layer_Info['inputs_layersName']
                inputlen = len(inp_layersName)
                inp_layersName = " ".join(x for x in inp_layersName)
                if layer_Info['layer_Type'] == 'Convolution':
                    layername = layer_Info['cur_layer_Name']
                    c_inputs    = layer_Info['number_of_Input_Feature_Channels']
                    c_output = layer_Info['number_of_Output_Feature_Channels']
                    kernel_w = layer_Info['kernel_Width']
                    kernel_h = layer_Info['kernel_Height']
                    dilation_w = layer_Info['dilation_Size_X']
                    dilation_h = layer_Info['dilation_Size_Y']
                    stride_w = layer_Info['stride_Size_X']
                    stride_h = layer_Info['stride_Size_Y']
                    pad_w = layer_Info['padding_Size_X']
                    pad_h = layer_Info['padding_Size_Y']
                    bias_term = layer_Info['if_Use_Bias']
                    weight_data_size0 = len(layer_Info['weights_Numpy_NCHW_F32'])
                    convtypeName = 'Convolution'
                    convgroup = layer_Info['num_of_Group']
                    weight_data_size = c_inputs * kernel_h * kernel_w * c_output
                    if convgroup > 1:
                        convtypeName = 'ConvolutionDepthWise'
                        weight_data_size = c_inputs * kernel_h * kernel_w
                        fparam.write(
                            '%s %32s  %32d %d  %32s %s 0=%d 1=%d 11=%d 2=%d 12=%d 3=%d 13=%d 4=%d 14=%d 5=%d 6=%d 7=%d\n' %
                            (convtypeName, layername, inputlen, 1, inp_layersName, layername, c_output, kernel_w,
                             kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_w, pad_h, bias_term,
                             weight_data_size,convgroup))
                    else:
                        fparam.write('%s %32s  %32d %d  %32s %s 0=%d 1=%d 11=%d 2=%d 12=%d 3=%d 13=%d 4=%d 14=%d 5=%d 6=%d\n' %
                            (convtypeName, layername, inputlen, 1, inp_layersName, layername, c_output, kernel_w,
                             kernel_h, dilation_w,dilation_h, stride_w, stride_h, pad_w, pad_h, bias_term, weight_data_size))


                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([ str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                        ####################write bin file for weight and bias##########
                    with open(self.binFilePath, "ab+") as fbin:
                        tag_qz = np.int32(0)
                        fbin.write(tag_qz.tobytes())
                        weight = layer_Info['weights_Numpy_NCHW_F32']
                        weightbin = weight.astype('float32').tobytes()
                        fbin.write(weightbin)
                        if bias_term != 0:
                            bias   = layer_Info['bias_Numpy_F32']
                            fbin.write(bias.astype('float32').tobytes())
                        pass
                    pass

                elif layer_Info['layer_Type'] == 'BatchNorm':
                    layername = layer_Info['cur_layer_Name']
                    channels = layer_Info['number_of_InOut_Feature_Channels']
                    fparam.write('%s %32s  %32d %d  %32s %s 0=%d\n' % ('BatchNorm', layername, inputlen, 1, inp_layersName,layername, channels))
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write(
                            '%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                    pass
                    ####################write bin file for weight and bias##########
                    with open(self.binFilePath, "ab+") as fbin:
                        weight_data = layer_Info['slope_data'] #scale
                        mean_data = layer_Info['mean_data']
                        var_data  = layer_Info['var_data']
                        bias_data = layer_Info['bias_data'] #B
                        fbin.write(weight_data.astype('float32').tobytes())
                        fbin.write(mean_data.astype('float32').tobytes())
                        fbin.write(var_data.astype('float32').tobytes())
                        fbin.write(bias_data.astype('float32').tobytes())
                        pass
                    pass
                elif layer_Info['layer_Type'] == 'MLinear':
                    layername = layer_Info['cur_layer_Name']

                    fparam.write('%s %32s  %32d %d  %32s %s 0=8 1=1 2=512\n' % ('InnerProduct', layername, inputlen, 1, inp_layersName,layername))
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write(
                            '%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                    pass
                    ####################write bin file for weight and bias##########
                    with open(self.binFilePath, "ab+") as fbin:
                        weight_data = layer_Info['weights_data'] #scale
                        bias_data = layer_Info['bias_data'] #B
                        tag_qz = np.int32(0)
                        fbin.write(tag_qz.tobytes())
                        fbin.write(weight_data.astype('float32').tobytes())
                        fbin.write(bias_data.astype('float32').tobytes())
                        pass
                    pass
                elif layer_Info['layer_Type'] == 'ReLU':
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    fparam.write('%s %32s  %32d %d  %32s %s\n' % ('ReLU', layername, inputlen, 1, inp_layersName, layername))
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write(
                            '%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                elif layer_Info['layer_Type'] == 'Pooling':  # split

                    layername = layer_Info['cur_layer_Name']
                    poolType = layer_Info['pooling_Type']
                    kernel_w = layer_Info['kernel_Height']
                    kernel_h = layer_Info['kernel_Width']
                    stride_w = layer_Info['stride_Size_X']
                    stride_h = layer_Info['stride_Size_Y']
                    pad_left = layer_Info['padding_Size_X']
                    pad_right = layer_Info['padding_Size_X']
                    pad_top = layer_Info['padding_Size_Y']
                    pad_bottom = layer_Info['padding_Size_Y']
                    global_pooling = 0 #layer_Info['']
                    pad_mode = 1#layer_Info['']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    fparam.write('%s %32s  %32d %d  %32s %s 0=%d 1=%d 11=%d 2=%d 12=%d 3=%d 13=%d 14=%d 15=%d 5=%d\n' % ('Pooling', layername, inputlen, 1, inp_layersName, layername, poolType,kernel_w,kernel_h,stride_w,stride_h,
                        pad_left,pad_right,pad_top,pad_bottom,pad_mode))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                elif layer_Info['layer_Type'] == 'Concat': #split
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    dim = layer_Info['Concat_Dim']
                    fparam.write('%s %32s  %32d %d  %32s %s 0=%d\n' % ('Concat', layername, inputlen, 1, inp_layersName, layername,dim))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                elif layer_Info['layer_Type'] == 'Upsample':#split
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Upsample', layername, 1, 1, inp_layersName, layername))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount

                    pass
                elif layer_Info['layer_Type'] == 'Add':#split
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    fparam.write('%s %32s  %32d %d  %32s %s 0=0\n' % ('BinaryOp', layername, 2, 1, inp_layersName, layername))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                    pass
                elif layer_Info['layer_Type'] == 'Permute':#split
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    dims = layer_Info['dims']
                    dim_params = "0=5"
                    #dim_params =  " ".join([(str(dims.index(sn))+'=' +str(sn)) for sn in dims])
                    fparam.write('%s %32s  %32d %d  %32s %s %s\n' % ('Permute', layername, 1, 1, inp_layersName, layername,dim_params))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                    pass
                elif layer_Info['layer_Type'] == 'Reshape':#split
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    tshap = layer_Info['shap'] #(4,4,1) is bug ,cause

                    shap_params = " ".join([(str(tshap.index(sn))+'=' +str(sn)) for sn in tshap])
                    fparam.write('%s %32s  %32d %d  %32s %s %s\n' % ('Reshape', layername, 1, 1, inp_layersName, layername,shap_params))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                    pass
                elif layer_Info['layer_Type'] == 'Softmax':#split
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    paramDim = layer_Info['softmax_Dim']

                    fparam.write('%s %32s  %32d %d  %32s %s 0=%s\n' % ('Softmax', layername, 1, 1, inp_layersName, layername,paramDim))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                    pass
                elif layer_Info['layer_Type'] == 'ReduceMean':
                    layername = layer_Info['cur_layer_Name']
                    ncnn_layer_Num += 1
                    ncnn_blob_Num += 1
                    rparams = layer_Info['params'] #(4,4,1) is bug ,cause

                    r_prams = " ".join([(str(rparams.index(sn))+'=' +str(sn)) for sn in rparams])
                    fparam.write('%s %32s  %32d %d  %32s %s %s\n' % ('ReduceMean', layername, 1, 1, inp_layersName, layername,r_prams))
                    pass
                    #######split pooling into several outputs
                    if layer_Info['splitCnt'] > 1:
                        outnms = layer_Info['outputs_layersName']
                        splitCount = len(outnms)
                        splitNames = " ".join([str(sn) for sn in outnms])
                        fparam.write('%s %32s  %32d %d  %32s %s\n' % ('Split', layername, 1, splitCount, layername, splitNames))
                        ncnn_layer_Num += 1
                        ncnn_blob_Num += splitCount
                    pass
                else:
                    print("Type info is new ,no suitable ncnn layer for compared...")
        fparam.close()
        fbin.close()

        ####rewrite split layer counts###
        ltxt = str('%d %d' % (ncnn_layer_Num, ncnn_blob_Num))
        self.reWrite_locLine(self.paramFilePath,2,ltxt,False,False)

        return True