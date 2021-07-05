# Question Answering System

In an **Ordering System**, the conversation is quite important to realize the users said and do a correct reply. In this project, we are using **BERT** to construct this Question Answering System which trained by the Chinese McDonald and MosBurger menu.

## Menu Generator

To generate the training data, we constructed the python script with menu and some questions which like below sample for Chinese McDonald Menu.

* Menu

    ```py
    main_meals = {
        "大麦克": (72, 16),
        "双层牛肉吉事堡": (62, 29),
        "嫩煎鸡腿堡": (82, 40),

        ...
    }
    ```
* Questions

    ```py
    questions = [
        "%s多少钱？",
        "请问%s价格？",
        "您好我要一份%s，这样价格为多少？",
        "我要一份%s，这样多少？"
    ]
    ```

By this menu generator, the users can just run the script,  `mc_menu_generator.py` or `mos_menu_generator.py`, to get the training data.

```sh
cd menu-generator
python mc_menu_generator.py
```

or

```sh
cd menu-generator
python mos_menu_generator.py
```

The generated training data, `test_1.json`, will like below with specific ID. (The `mos_menu_generator.py` will generate the `mos_test_1.json`.)

```json
"question": "大麦克多少钱？", 
"id": "MCDONALDS_168_QUERY_0", 
"answers": [
    {
        "text": "72元", 
        "answer_start": 16
    }
]
```

And it also generate the `paragraph.txt` for demo.

### Own Generator

If you want to enable the specific conversation or another menu, you can based on `mc_menu_generator.py` or `mos_menu_generator.py` to meet your requirement. In these two menu generator scripts, the meal is defined with the price and position(index) of answer, a reply, in context, `_set["context"]`. However, the index is not just from the answer in context, it needs to treate the all continuous number as one to meet the token design in BERT. Therefore, you need to care the all numbers in context get the correct position of the answer start to make the training data correct.

## Train a BERT to Support the Own Dataset

After generated the training data, you can based on the **CMRC2018** dataset with Chinese BERT pre-trained model to fine tune your Question Answering system for conversation. 

* Preparation
    
    Please check [Preparation section](https://github.com/FengYen-Chang/Chinese.BERT.OpenVINO#preparation) on repo., Chinese.BERT.OpenVINO to download the distilled bert model and follow below command to download cmrc2018 submodules.
    
    ```sh
    git submodule update --init ./extension/cmrc2018
    ```
    
    Install `tensorflow` version `1.15.0`.
    
    ```sh
    pip install tensorflow==1.15.0
    ```

* Fine Tuning the BERT Model for Conversation

    After preparation, we can use the pretrained and distilled BERT model and dataset, CMRC2018, with generated training data to fine tune the model.
    
    > In this repo., I am using `Roberta-wwm-ext-large-3layers-distill, Chinese`.

    ```sh
    cd extension/cmrc2018/baseline

    export PATH_TO_BERT=/path/to/distilled/bert/model/3layers_large
    export DATA_DIR=/path/to/generatd/dataset
    export OUTPUT_DIR=/path/to/save/result/and/tuned_model

    python run_cmrc2018_drcd_baseline.py                    \
        --vocab_file=${PATH_TO_BERT}/vocab.txt              \
        --bert_config_file=${PATH_TO_BERT}/bert_config.json \
        --init_checkpoint=${PATH_TO_BERT}/bert_model.ckpt   \
        --do_train=True                                     \
        --train_file=${DATA_DIR}/test_1.json                \
        --do_predict=True                                   \
        --predict_file=${DATA_DIR}/test_1.json              \
        --train_batch_size=32                               \
        --num_train_epochs=40                               \
        --max_seq_length=256                                \
        --doc_stride=128                                    \
        --learning_rate=3e-5                                \
        --save_checkpoints_steps=1000                       \
        --output_dir=${OUTPUT_DIR}                          \
        --do_lower_case=False                               \
        --use_tpu=False
    ```

    > If you want to evaluate the result, please check this [page](https://github.com/FengYen-Chang/Chinese.BERT.OpenVINO#evaluate-the-fine-tuning-result).

* Export PB Model

    ```sh
    export PATH_TO_BERT=/path/to/distilled/bert/model/3layers_large
    export DATA_DIR=/path/to/generatd/dataset
    export OUTPUT_DIR=/path/to/save/result/and/tuned_model
    export FINE_TUNED_CHECK_POINT=${OUTPUT_DIR}/model.ckpt-xxx

    python export_pb.py                                     \
        --vocab_file=${PATH_TO_BERT}/vocab.txt              \
        --bert_config_file=${PATH_TO_BERT}/bert_config.json \
        --init_checkpoint=${FINE_TUNED_CHECK_POINT}         \
        --do_train=True                                     \
        --train_file=${DATA_DIR}/test_1.json                \
        --do_predict=True                                   \
        --predict_file=${DATA_DIR}/test_1.json              \
        --train_batch_size=32                               \
        --num_train_epochs=40                               \
        --max_seq_length=256                                \
        --doc_stride=128                                    \
        --learning_rate=3e-5                                \
        --save_checkpoints_steps=1000                       \
        --output_dir=${OUTPUT_DIR}                          \
        --do_lower_case=False                               \
        --use_tpu=False
    ```
    

* Convert the `.pb` model to Intermediate Representation (IR)

    Please use below command to run the Model Optimizer for those converted `.pb` model to IR model.

    ```sh
    export MODEL_DIR=/path/to/converted/pb/model
    export OUTPUT_DIR=/path/to/converted/IR/model

    python mo.py 
        --input_model=${MODEL_DIR}/inference_graph.pb                                                       \
  	    --input "IteratorGetNext:0{i32}[1 256],IteratorGetNext:1{i32}[1 256],IteratorGetNext:3{i32}[1 256]" \
  	    --disable_nhwc_to_nchw                                                                              \
        -o ${OUTPUT_DIR}
    ```
