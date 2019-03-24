import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  ###make sure graph py could find your gv bin
import sys
import platform
import graphviz as gv
from string import Template
###########################################################################
Attrib_Dicts = {
    'Conv':         dict(style='filled',align='top',fontsize='26',ranksep='0.1',height='0.5',color='red2',shape='circle'),
    'BatchNorm':    dict(style='filled',align='top',fontsize='22',ranksep='0.1',height='0.1',color='blue2',shape='box'),
    'Default':      dict(style='filled',align='top',fontsize='22',ranksep='0.1',height='0.1',shape='box'),
}
###########################################################################
class  TorchGraph_Pen(object):
    def __init__(self, in_LayerDict, out_Path, gtype = 0, gnode_attr = {}):
        self.inSrcDict = in_LayerDict
        self.outputPath  = out_Path
        self.graph = gv.Digraph(format='png', node_attr=Attrib_Dicts['Default'])

    def __str__(self):
        return str("gv-graph")


    #############color='red2',shape='circle',height='.5'###########
    def makeNodes(self,layerInfo,cur_node_label='',node_attrib={}):
        in_nodes = layerInfo['inputs_layersName']
        cur_node  = layerInfo['cur_layer_Name']

        ###############add node ,edge into graph##############
        self.graph.node(cur_node,cur_node_label,node_attrib)
        edge_label = 'branch'
        edge_attri = ''
        if len(in_nodes) > 1:
            edge_label = '%d-inBranches' %(len(in_nodes))

        for in_node in in_nodes:
            self.graph.edge(in_node, cur_node, edge_label,edge_attri)

        return

    def makeLayer(self):
        code = Template('''subgraph cluster_$index {\n   
            color=white;\n   
            node [style=solid,color=$color];\n    
            $nodesB$nodesM$nodesE    label = "layer $name($node_num)";\n    
            }\n    
            ''')
        if len(self.inSrcDict) <= 1:
            print("no right dicts for drawing")
        ###############give a input node###########

        for layer_Info in self.inSrcDict:
            layer_Type = layer_Info['layer_Type']
            layername = layer_Info['cur_layer_Name']
            node_label = ''
            node_attrib = {}
            if layer_Type == 'Convolution':
                c_inputs = layer_Info['number_of_Input_Feature_Channels']
                c_output = layer_Info['number_of_Output_Feature_Channels']
                node_label += layername
                node_label += "" "\n" ""
                node_label += "" "inChannle:{}""".format(c_inputs)
                node_label += "" "\n" ""
                node_label += "" "outChannle:{}""".format(c_output)
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Conv']
            elif layer_Type == 'BatchNorm':
                channels = layer_Info['number_of_InOut_Feature_Channels']
                node_label += layername
                node_label += "" "\n" ""
                node_label += "" "BChannle:{}""".format(channels)
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['BatchNorm']
            elif layer_Type == 'ReLU':
                node_label += layername
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            elif layer_Info['layer_Type'] == 'Pooling':
                poolType = layer_Info['pooling_Type']
                node_label += layername
                node_label += "" "\n" ""
                node_label += "" "Pool:{}""".format(poolType)
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            elif layer_Info['layer_Type'] == 'Concat':
                inNum = len(layer_Info['inputs_layersName'])
                node_label += layername
                node_label += "" "\n" ""
                node_label += "" "In:{}""".format(inNum)
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            elif layer_Info['layer_Type'] == 'Upsample':
                node_label += layername
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            elif layer_Info['layer_Type'] == 'Permute':
                node_label += layername
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            elif layer_Info['layer_Type'] == 'Reshape':
                node_label += layername
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            elif layer_Info['layer_Type'] == 'Add':
                node_label += layername
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            elif layer_Info['layer_Type'] == 'Softmax':
                node_label += layername
                node_label += "" "\n" ""
                node_attrib = Attrib_Dicts['Default']
            else:
                print("Attention!!!!! some layer type not exsit!!!")
            pass
            self.makeNodes(layer_Info,node_label,node_attrib)
        pass
        self.graph.render(self.outputPath)
        gvPath = self.outputPath + '.gv'
        self.save_graphFile(self.graph.source, gvPath)
        return

    def save_graphFile(self, str, path):
        print (path)
        f = open(path,'w')
        f.write(str)
        f.close()
##########################################################################
