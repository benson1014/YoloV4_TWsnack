?$	m?%?ӣT@ ?1(??W@??i?????!'?
?0e@	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsD?b?5d@1?A?"L?U@I?Dׅ&R@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??i?????1???^a???I<L??????"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?h>?n??1?'eRC??I?@+0du??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'?
?0e@?%?h@1??_??2d@I,??f*@*	?A`e?A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???фq@!??ϭ?X@)???фq@1??ϭ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchB???ϝ??!\uX????)B???ϝ??1\uX????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?%?<??!?(?'???)?1?	????1?q?-rx?:Preprocessing2F
Iterator::Modelrn??y??!8^????)	3m??Js?1?q??a?[?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?_???q@!گg??X@)?fh<q?1j?U?]?X?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?23.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?|
?v8@Q?`}?y?R@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?%?h???%?h??!?%?h@	!       "$	[Υ??.O@@??KJ?S@?'eRC??!??_??2d@*	!       2	!       :$	?
E??3@?#]?<?A@<L??????!?Dׅ&R@B	!       J	!       R	!       Z	!       b	!       JGPUb q?|
?v8@y?`}?y?R@?"d
9gradient_tape/model_1/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?#??;a??!?#??;a??0"g
;gradient_tape/model_1/conv2d_59/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterI???`r??!?Jg?i??0"h
>gradient_tape/model_1/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3?Ò?7ه?!???5???"f
:gradient_tape/model_1/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterDۮ??/??!X?5&7??0"g
;gradient_tape/model_1/conv2d_76/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW?????!Ȣx??0"h
<gradient_tape/model_1/conv2d_104/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!??RC?ڱ?0"g
;gradient_tape/model_1/conv2d_73/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????K???!?}?8x??0"h
<gradient_tape/model_1/conv2d_106/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????J??!?.̐??0"h
<gradient_tape/model_1/conv2d_108/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6??N????!U?΀??0"R
,model_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3???????!??%*????Q      Y@Yz??3"???a??1wˍX@q???5?
@y!???>?S?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?23.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 