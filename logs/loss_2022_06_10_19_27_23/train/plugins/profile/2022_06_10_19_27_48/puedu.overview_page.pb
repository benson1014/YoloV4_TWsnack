?$	??@?#??:????????D?????!??? ???	!       "^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??? ???1q??sC??IV????/??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsc???J??1eު?PM??I??&?????"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?ra???1?&?|???I??҇.???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails>=?e?Y??1?h㈵???I?(5
??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?D?????1C?8
q?Is?]?????"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???f???1?&?|???I8???LM??*	???S??K@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchgs?69??!8???H@)gs?69??18???H@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism4?s?륩?!??~??~V@)2t????1????<D@:Preprocessing2F
Iterator::Model?ի?耬?!      Y@)B?????v?1?	?	$@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?91.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI 2z??V@Q?no.?:!@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	t??d? ????	???C?8
q?!q??sC??*	!       2	!       :$	?????????PRV????s?]?????!V????/??B	!       J	!       R	!       Z	!       b	!       JGPUb q 2z??V@y?no.?:!@?".
IteratorGetNext/_44_Recv?h?????!?h?????"5
model_1/conv2d_1/Conv2DConv2D?n??Eo??!??????0"g
;gradient_tape/model_1/conv2d_73/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??>??4??!F???m6??0"g
;gradient_tape/model_1/conv2d_76/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterOK?-4$??!?M??ڳ?0"h
<gradient_tape/model_1/conv2d_104/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter%y5?? ??!???q{??0"h
<gradient_tape/model_1/conv2d_106/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?&??}???!?a[-???0"h
<gradient_tape/model_1/conv2d_108/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?雯???!I?? ????0"5
model_1/conv2d_5/Conv2DConv2D???q5??!??!P???0"5
model_1/conv2d_8/Conv2DConv2D?Y???4??!??S^???0"3
model_1/conv2d/Conv2DConv2D/?]?۞??!v?I?g??0Q      Y@Y?3U"??@abV???X@q???????y̘踕
Q?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?91.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 