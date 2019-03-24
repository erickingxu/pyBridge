# pyBridge
This a bridge for converting torch,and other AI training framework to C++ speed up infer library,like NCNN,and ect...

Firstly,
add pyBridge to your main pytorch project ,and refer to transferMain.py for details...

Secondly,
best way for converting pytorch to ncnn is hooking principle in torch, so essentially，every function in pytorch ,this hook_walker
will can catch and convert it.

Third,
some graph layer was added for gv format refixing ,especially for graphviz.Maybe prime learning in AI field ,can see the entity
of whole network when network is very big....

Fourth,
some other networks like caffe and mxnet convertings will be update in some day ,or some pretty babies can help for you,that is
very appricated....

Help yourself.....
