$	m?%?ӣT@ ?1(??W@??i?????!'?
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
?v8@y?`}?y?R@