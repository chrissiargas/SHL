?  *	?A`e?/A2?
[Iterator::Root::Prefetch::BatchV2::ShuffleAndRepeat::MemoryCacheImpl::FlatMap[0]::Generator ?^e?q@!S2?$?X@)?^e?q@1S2?$?X@:Preprocessing2j
3Iterator::Root::Prefetch::BatchV2::ShuffleAndRepeat z?I|??q@!?R=?~?X@)iƢ????1;?9?R???:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2g~5G?q@!??"?.?X@)àL?????1!D?*????:Preprocessing2{
DIterator::Root::Prefetch::BatchV2::ShuffleAndRepeat::MemoryCacheImpl ˞6'?q@!j??u{?X@)????kz??1ݫ?_i???:Preprocessing2?
MIterator::Root::Prefetch::BatchV2::ShuffleAndRepeat::MemoryCacheImpl::FlatMap ????׏q@!t???3?X@)7???-??1I?[v??:Preprocessing2O
Iterator::Root::Prefetch??Aȗ??!?%?P֑??)??Aȗ??1?%?P֑??:Preprocessing2w
@Iterator::Root::Prefetch::BatchV2::ShuffleAndRepeat::MemoryCache ????q@!??]??X@)*8? "5??1?&?????:Preprocessing2E
Iterator::Root?z?L?x??!@????)#h?$???1 ?+?v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@qb??~@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JDESKTOP-2COUPRM: Failed to load libcupti (is it installed and accessible?)