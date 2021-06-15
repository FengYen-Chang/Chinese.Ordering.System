# Speech Recognition


In an **Ordering System**, the speech recognition function is a key function to recognize what user said to make it as text for language model, bert. In here, This repository is using the **DeepSpeech** to enable this function.

In this page, it shows the all steps to enable the Chinese base DeepSpeech by OpenVINO.

### Preparation

* Download the Chinese Base model from DeepSpeech [page](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models-zh-CN.pbmm).

  ```sh
  wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models-zh-CN.pbmm
  ```
  
### Convert the model from `.pbmm` to `.pb`

  ```sh
  export MODEL_DIR=/path/to/IR/model/directory
  
  cd model-conversion/mozilla-deepspeech-0.9.3-zh-CN
  python pbmm_to_pb.py deepspeech-0.9.3-models-zh-CN.pbmm deepspeech-0.9.3-models-zh-CN.pb
  ```
  
> In `pbmm_to_pb.py` file, you can see the below code:
> ```py
> def del_node(node, temp_node):
>     node.attr['value'].tensor.string_val[:] = temp_node.attr['value'].tensor.string_val[:]
> 
> def remove_metadata_alphabet(graph_def):
>     """
>     remove alphabet metadata to avoid upexpected ascii code. 
>     """
>     temp_node = None
>     for node in graph_def.node:
>         if node.name == 'metadata_language':
>             temp_node = node
>     
>     for node in graph_def.node:
>         if node.name == 'metadata_alphabet':
>             del_node(node, temp_node)
> ```
> Because the `.pbmm` model saved several information, such as `metadata_alphabet`, and this `metadata_alphabet` will cause the error during convert the model to IR as the unknown utf-8 code. Therefore, we replaced the data by above function to avoid the error.

### Convert the `.pb` model to Intermediate Representation (IR)

```sh
export MODEL_DIR=/path/to/converted/pb/model

python mo.py                                                  \
  --input_model=${MODEL_DIR}/deepspeech-0.9.3-models-zh-CN.pb \
  --freeze_placeholder_with_value=input_lengths->[16]         \
  --input=input_node,previous_state_h,previous_state_c        \
  --input_shape=[1,16,19,26],[1,2048],[1,2048]                \
  --disable_nhwc_to_nchw                                      \
  --output=logits,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1
```
