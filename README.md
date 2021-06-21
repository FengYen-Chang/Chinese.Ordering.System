# Chinese Ordering System 

**Notice:**
 * **This reposiotry still under develop.**
    * Still debug the Chinese language model based on the kenlm Version 5 for speech recognize model.
    * Will upload the model conversation method of Chinese language model and speech recognize model.
 * Currently, this reposiotry is using `ds_ctcdecoder`version `0.9.3` to replace `ctcdecode_numpy`.
   ```sh
   pip install ds-ctcdecoder=0.9.3
   ```

 * Please pay attention to the speech recognize model of license, **Mozilla Public License 2.0**.

This is an ordering system wihch based on the 
* [Speech Recognition System](./SpeechRecognition.md),
* [Question Answering System](./QuestionAnswering.md) and 
* [Voice Synthesize System](./VoiceSynthesize.md) 

for Chinese. By this system, user can order a meal with voice and the ordering system will reply the price.

# Enable Each Components

Please check below pages to enable it. Based on these pages, you can get the converted IR model and you can follow next section to execute it with converted IR model.

* [Speech Recognition System](./SpeechRecognition.md)
* [Question Answering System](./QuestionAnswering.md)
* [Voice Synthesize System](./VoiceSynthesize.md) 

# Run each function

You can use below links to run each function or scroll down this page.

* [Execute Ordering System](#execute-ordering-system)
* [Execute Speech Recognize System](#execute-speech-recognize-system)
* [Execute Question Answering System](#execute-question-answering-system)
* [Execute Voice Synthesize System](#execute-voice-synthesize-system)
* [Execute Question Answering System with Voice Synthesize System](#execute-question-answering-system-with-voice-synthesize-system)

### Execute Ordering System

Runs the `ordering_system_demo_pipeline.py` for demo. Running the application with the -h option yields the following usage message:

```sh
usage: ordering_system_demo_pipeline.py [-h] -m_mel MODEL_MEL -m_mg
                                        MODEL_MELGAN -m_d MODEL_DECODER -m_e
                                        MODEL_ENCODER -m_dp
                                        MODEL_DURATION_PREDICTOR -m_b
                                        MODEL_BERT -m_a MODEL_AUDIO
                                        [-L FILENAME] -i INPUT -p NAME -para
                                        PARAGRAPH -v_b VOCAB_BERT -v_p
                                        VOCAB_PINYIN [-d DEVICE] [-b N] [-c N]
                                        [--max-seq-length MAX_SEQ_LENGTH]
                                        [--doc-stride DOC_STRIDE]
                                        [--max-query-length MAX_QUERY_LENGTH]
                                        [-mal MAX_ANSWER_LENGTH]
                                        [-nbest NUM_OF_BEST_SET] [-l FILENAME]

Ordering System demo

Options:
  -h, --help            Show this help message and exit.
  -m_mel MODEL_MEL, --model-mel MODEL_MEL
                        Required. Path to an .xml file with a trained mel
                        model.
  -m_mg MODEL_MELGAN, --model-melgan MODEL_MELGAN
                        Required. Path to an .xml file with a trained mel gan
                        model.
  -m_d MODEL_DECODER, --model-decoder MODEL_DECODER
                        Required. Path to an .xml file with a trained decoder
                        model.
  -m_e MODEL_ENCODER, --model-encoder MODEL_ENCODER
                        Required. Path to an .xml file with a trained encoder
                        model.
  -m_dp MODEL_DURATION_PREDICTOR, --model-duration_predictor MODEL_DURATION_PREDICTOR
                        Required. Path to an .xml file with a trained duration
                        predictor model.
  -m_b MODEL_BERT, --model-bert MODEL_BERT
                        Required. Path to an .xml file with a trained bert
                        model.
  -m_a MODEL_AUDIO, --model-audio MODEL_AUDIO
                        Required. Path to an .xml file with a trained audio
                        model.
  -L FILENAME, --lm FILENAME
                        path to language model file (optional)
  -i INPUT, --input INPUT
                        Required. Path to a 16k .wav file for audio model.
  -p NAME, --profile NAME
                        Choose pre/post-processing profile: mds06x_en for
                        Mozilla DeepSpeech v0.6.x, mds07x_en or mds08x_en for
                        Mozilla DeepSpeech v0.7.x/x0.8.x, mds09x_cn for
                        Mozilla DeepSpeech v0.9.x Chinese Model, other:
                        filename of a YAML file (required)
  -para PARAGRAPH, --paragraph PARAGRAPH
                        Required. Path to a .txt file w/ paragraph for bert.
  -v_b VOCAB_BERT, --vocab-bert VOCAB_BERT
                        Required. Path to vocabulary file for bert model.
  -v_p VOCAB_PINYIN, --vocab-pinyin VOCAB_PINYIN
                        Required. Path to pinyin vocabulary file for TTS
                        model.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
  -b N, --beam-width N  Beam width for beam search in CTC decoder (default
                        500)
  -c N, --max-candidates N
                        Show top N (or less) candidates (default 1)
  --max-seq-length MAX_SEQ_LENGTH
                        Optional. The maximum total input sequence length
                        after WordPiece tokenization.
  --doc-stride DOC_STRIDE
                        Optional.When splitting up a long document into
                        chunks, how much stride to take between chunks.
  --max-query-length MAX_QUERY_LENGTH
                        Optional. The maximum number of tokens for the
                        question. Questions longer than this will be truncated
                        to this length.
  -mal MAX_ANSWER_LENGTH, --max_answer_length MAX_ANSWER_LENGTH
                        Optional. The maximum length of an answer that can be
                        generated.
  -nbest NUM_OF_BEST_SET, --num_of_best_set NUM_OF_BEST_SET
                        Optional. The number for n-best predictions to
                        generate the final result
  -l FILENAME, --cpu_extension FILENAME
                        Optional. Required for CPU custom layers. MKLDNN
                        (CPU)-targeted custom layers. Absolute path to a
                        shared library with the kernels implementations.
```

#### Running Inference

* Run inference:
    
    ```sh
    export MODEL_DIR=/path/to/IR/model/directory
    export VOCAB_DIR=/path/to/vocabulary/directory
    
    python ordering_system_demo_pipeline.py                       \
        -m_mel  ${MODEL_DIR}/mel.xml                              \
        -m_mg   ${MODEL_DIR}/melgan.xml                           \
        -m_d    ${MODEL_DIR}/decoder.xml                          \
        -m_e    ${MODEL_DIR}/encoder.xml                          \
        -m_dp   ${MODEL_DIR}/duration_predictor.xml               \
        -m_b    ${MODEL_DIR}/bert.xml                             \
        -m_a    ${MODEL_DIR}/deepspeech-0.9.3-models-zh-CN.xml    \
        -p mds09x_cn                                              \
        -para ${PARAGRAPH_FILE}                                   \
        -i audio/sample.wav                                       \
        -v_b ${VOCAB_DIR}/vocab_bert.txt                          \
        -v_p ${VOCAB_DIR}/vocab_pinyin.txt                        \
        -L ${MODEL_DIR}/deepspeech-0.9.3-models-zh-CN.scorer
    ```
    
   * Example for paragraph, `mc_paragraph.txt`.
      
      ```sh
      麦当劳目前的餐点有：大麦克价格为72元、双层牛肉吉事堡价格为62元、嫩煎鸡腿堡价格为82元、麦香鸡价格为44元、麦克鸡块(6块)价格为60元、麦克鸡块(10块)价格为100元、劲辣鸡腿堡价格为72元、麦脆(2块)价格为110元、麦脆鸡翅(2块)价格为90元、黄金起司猪排堡价格为52元、麦香鱼价格为44元、烟熏鸡肉长堡价格为74元、姜烧猪肉长堡价格为74元、BLT 安格斯黑牛堡价格为109元、BLT 辣脆鸡腿堡价格为109元、BLT 嫩煎鸡腿堡价格为109元、蕈菇安格斯黑牛堡价格为119元、凯萨脆鸡沙拉价格为99元和义式烤鸡沙拉价格为99元。
      ```
      
      > The content of the `mc_paragraph.txt` is from Bert's training data.

* Output:

    ```sh
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 20.84it/s]
    [INFO] 2021-06-18 14:59:58,559 Load 1 examples
    Content:  麦当劳目前的餐点有：大麦克价格为72元、双层牛肉吉事堡价格为62元、嫩煎鸡腿堡价格为82元、麦香鸡价格为44元、麦克鸡块(6块)价格为60元、麦克鸡块(10块)价格为100元、劲辣鸡腿堡价格为72元、麦脆鸡腿(2块)价格为110元、麦脆鸡翅(2块)价格为90元、黄金起司猪排堡价格为52元、麦香鱼价格为44元、烟熏鸡肉长堡价格为74元、姜烧猪肉长堡价格为74元、BLT安格斯黑牛堡价格为109元、BLT辣脆鸡腿堡价格为109元、BLT嫩煎鸡腿堡价格为109元、蕈菇安格斯黑牛堡价格为119元、凯萨脆鸡沙拉价格为99元和义式烤鸡沙拉价格为99元。
    Question:  大麦克多少钱
    Building prefix dict from the default dictionary ...
    [DEBUG] 2021-06-18 14:59:58,609 Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    [DEBUG] 2021-06-18 14:59:58,610 Loading model from cache /tmp/jieba.cache
    Loading model cost 0.422 seconds.
    [DEBUG] 2021-06-18 14:59:59,031 Loading model cost 0.422 seconds.
    Prefix dict has been built successfully.
    [DEBUG] 2021-06-18 14:59:59,031 Prefix dict has been built successfully.
    /usr/lib/python3/dist-packages/apport/report.py:13: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import fnmatch, glob, traceback, errno, sys, atexit, locale, imp
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    [INFO] 2021-06-18 14:59:59,853 Generated tts.wav.
    [INFO] 2021-06-18 14:59:59,853 This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
    ```
    
    > Once the process is done, we can find the `tts.wav` in the working directory.
    > For this sample, you can hear the `七十二元` in the `tts.wav`

### Execute Speech Recognize System

For speech recognize system, you can run `speech_recognition_demo.py` which is from [OMZ](https://github.com/openvinotoolkit/open_model_zoo) to execute it.

Running the application with the -h option yields the following usage message:

```sh
python speech_recognition_demo.py -h
usage: speech_recognition_demo.py [-h] -i FILENAME [-d DEVICE] -m FILENAME
                                  [-L FILENAME] -p NAME [-b N] [-c N]
                                  [-l FILENAME]

Speech recognition demo

optional arguments:
  -h, --help            show this help message and exit
  -i FILENAME, --input FILENAME
                        Path to an audio file in WAV PCM 16 kHz mono format
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on, for
                        example: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO. The
                        sample will look for a suitable IE plugin for this
                        device. (default is CPU)
  -m FILENAME, --model FILENAME
                        Path to an .xml file with a trained model (required)
  -L FILENAME, --lm FILENAME
                        path to language model file (optional)
  -p NAME, --profile NAME
                        Choose pre/post-processing profile: mds06x_en for
                        Mozilla DeepSpeech v0.6.x, mds07x_en or mds08x_en for
                        Mozilla DeepSpeech v0.7.x/x0.8.x, mds09x_cn for
                        Mozilla DeepSpeech v0.9.x Chinese Model, other:
                        filename of a YAML file (required)
  -b N, --beam-width N  Beam width for beam search in CTC decoder (default
                        500)
  -c N, --max-candidates N
                        Show top N (or less) candidates (default 1)
  -l FILENAME, --cpu_extension FILENAME
                        Optional. Required for CPU custom layers. MKLDNN
                        (CPU)-targeted custom layers. Absolute path to a
                        shared library with the kernels implementations.
```

#### Running Inference

* Run inference:
    
    ```sh
    export MODEL_DIR=/path/to/IR/model/directory
    
    python speech_recognition_demo.py                             \
        -m ${MODEL_DIR}/deepspeech-0.9.3-models-zh-CN.xml         \
        -p mds09x_cn                                              \
        -i audio/sample.wav                                       \
        -L ${MODEL_DIR}/deepspeech-0.9.3-models-zh-CN.scorer
    ```

* Output:
   
   ```sh
   Loading, including network weights, IE initialization, LM, building LM vocabulary trie, loading audio: 0.5604348049964756 s
   Audio file length: 1.9505 s
   MFCC time: 0.002272702055051923 s
   100%|████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 20.24it/s]
   RNN time: 0.29857360396999866 s
   Beam search time: 0.24717673601116985 s
   Overall time: 1.1114037550287321 s

   Transcription and confidence score:
   -42.48539733886719      大麦克多少钱
   ```


### Execute Question Answering System

For Question Answering System, please run below command first, 

```sh
git submodule update --init ./extension/Chinese.BERT.OpenVINO
```

,and check this [page](./extension/Chinese.BERT.OpenVINO/README.md)

### Execute Voice Synthesize System

For voice synthesize system, you can run `tts.py` to execute it.

Running the application with the -h option yields the following usage message:

```sh
python3 tts.py -h
usage: tts.py [-h] -m_mel MODEL_MEL -m_mg MODEL_MELGAN -m_d MODEL_DECODER -m_e
              MODEL_ENCODER -m_dp MODEL_DURATION_PREDICTOR -i INPUT
              [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -m_mel MODEL_MEL, --model_mel MODEL_MEL
                        Required. Path to an .xml file with a trained mel
                        model.
  -m_mg MODEL_MELGAN, --model_melgan MODEL_MELGAN
                        Required. Path to an .xml file with a trained mel gan
                        model.
  -m_d MODEL_DECODER, --model_decoder MODEL_DECODER
                        Required. Path to an .xml file with a trained decoder
                        model.
  -m_e MODEL_ENCODER, --model_encoder MODEL_ENCODER
                        Required. Path to an .xml file with a trained encoder
                        model.
  -m_dp MODEL_DURATION_PREDICTOR, --model_duration_predictor MODEL_DURATION_PREDICTOR
                        Required. Path to an .xml file with a trained duration
                        predictor model.
  -i INPUT, --input INPUT
                        Required. Path to a text which you want to make as
                        speech.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
```

#### Running Inference

* Run inference:

    ```sh
    export MODEL_DIR=/path/to/IR/model/directory

    python tts.py                                   \
        -m_mel  ${MODEL_DIR}/mel.xml                \
        -m_mg   ${MODEL_DIR}/melgan.xml             \
        -m_e    ${MODEL_DIR}/encoder.xml            \
        -m_d    ${MODEL_DIR}/decoder.xml            \
        -m_dp   ${MODEL_DIR}/duration_predictor.xml \
        -i 一百五十
    ```

* Output:

    ```sh
    [ INFO ] Creating Inference Engine
    [ INFO ] Loading mel network files:
        models/mel.xml
        models/mel.bin
    [ INFO ] Loading melgan network files:
        models/melgan.xml
        models/melgan.bin
    [ INFO ] Loading decoder network files:
        models/decoder.xml
        models/decoder.bin
    [ INFO ] Loading encoder network files:
        models/encoder.xml
        models/encoder.bin
    [ INFO ] Loading duration predictor network files:
        models/duration_predictor.xml
        models/duration_predictor.bin
    [ INFO ] Loading model to the plugin
    processing 一百五十
    Building prefix dict from the default dictionary ...
    [ DEBUG ] Building prefix dict from the default dictionary ...
    Dumping model to file cache /tmp/jieba.cache
    [ DEBUG ] Dumping model to file cache /tmp/jieba.cache
    Loading model cost 0.825 seconds.
    [ DEBUG ] Loading model cost 0.825 seconds.
    Prefix dict has been built successfully.
    [ DEBUG ] Prefix dict has been built successfully.
    /usr/lib/python3/dist-packages/apport/report.py:13: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
    import fnmatch, glob, traceback, errno, sys, atexit, locale, imp
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    [ INFO ] Generated tts.wav.
    [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
    ```

    > Once the process is done, we can find the `tts.wav` in the working directory.

### Execute Question Answering System with Voice Synthesize System

Please run `bert+tts.py` to execute it.

Running the application with the -h option yields the following usage message:

```sh
python bert+tts.py -h
usage: bert+tts.py [-h] -m_mel MODEL_MEL -m_mg MODEL_MELGAN -m_d MODEL_DECODER
                   -m_e MODEL_ENCODER -m_dp MODEL_DURATION_PREDICTOR -m_b
                   MODEL_BERT -i INPUT -v VOCAB [-d DEVICE]
                   [--max-seq-length MAX_SEQ_LENGTH] [--doc-stride DOC_STRIDE]
                   [--max-query-length MAX_QUERY_LENGTH] [-qn QUESTION_NUMBER]
                   [-mal MAX_ANSWER_LENGTH] [-nbest NUM_OF_BEST_SET]

Options:
  -h, --help            Show this help message and exit.
  -m_mel MODEL_MEL, --model_mel MODEL_MEL
                        Required. Path to an .xml file with a trained mel
                        model.
  -m_mg MODEL_MELGAN, --model_melgan MODEL_MELGAN
                        Required. Path to an .xml file with a trained mel gan
                        model.
  -m_d MODEL_DECODER, --model_decoder MODEL_DECODER
                        Required. Path to an .xml file with a trained decoder
                        model.
  -m_e MODEL_ENCODER, --model_encoder MODEL_ENCODER
                        Required. Path to an .xml file with a trained encoder
                        model.
  -m_dp MODEL_DURATION_PREDICTOR, --model_duration_predictor MODEL_DURATION_PREDICTOR
                        Required. Path to an .xml file with a trained duration
                        predictor model.
  -m_b MODEL_BERT, --model_bert MODEL_BERT
                        Required. Path to an .xml file with a trained bert
                        model.
  -i INPUT, --input INPUT
                        Required. Path to a json file w/ description and
                        question.
  -v VOCAB, --vocab VOCAB
                        Required. Path to vocablary file for bert model.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
  --max-seq-length MAX_SEQ_LENGTH
  --doc-stride DOC_STRIDE
  --max-query-length MAX_QUERY_LENGTH
  -qn QUESTION_NUMBER, --question_number QUESTION_NUMBER
  -mal MAX_ANSWER_LENGTH, --max_answer_length MAX_ANSWER_LENGTH
  -nbest NUM_OF_BEST_SET, --num_of_best_set NUM_OF_BEST_SET
```

#### Running Inference

* Input file `mc_menu.json`:

    ```json
    {
        "version": "v1.0",
        "data": [
            {
                "paragraphs": [
                    {
                        "id": "MCDONALDS_168",
                        "context": "麦当劳目前的餐点有：大麦克价格为72元、双层牛肉吉事堡价格为62元、嫩煎鸡腿堡价格为82元、麦香鸡价格为44元、麦克鸡块(6块)价格为60元、麦克鸡块(10块)价格为100元、劲辣鸡腿堡价格为72元、麦脆鸡腿(2块)价格为110元、麦脆鸡翅(2块)价格为90元、黄金起司猪排堡价格为52元、麦香鱼价格为44元、烟熏鸡肉长堡价格为74元、姜烧猪肉长堡价格为74元、BLT 安格斯黑牛堡价格为109元、BLT 辣脆鸡腿堡价格为109元、BLT 嫩煎鸡腿堡价格为109元、蕈菇安格斯黑牛堡价格为119元、凯萨脆鸡沙拉价格为99元和义式烤鸡沙拉价格为99元。",
                        "qas": [
                            {
                                "question": "大麦克多少钱？",
                                "id": "MCDONALDS_168_QUERY_0",
                                "answers": [
                                    {
                                        "text": "72元",
                                        "answer_start": 16
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    ```

* Run inference:

    ```sh
    export MODEL_DIR=/path/to/IR/model/directory

    python bert+tts.py                              \
        -m_mel  ${MODEL_DIR}/mel.xml                \
        -m_mg   ${MODEL_DIR}/melgan.xml             \
        -m_e    ${MODEL_DIR}/encoder.xml            \
        -m_d    ${MODEL_DIR}/decoder.xml            \
        -m_dp   ${MODEL_DIR}/duration_predictor.xml \
        -m_b    ${MODEL_DIR}/bert.xml               \
        -i mc_menu.json                             \
        -v vocab/vocab_bert.txt
    ```

* Output

    ```sh
    python bert+tts.py -m_mel models/mel.xml -m_mg models/melgan.xml -m_e models/encoder.xml -m_d models/decoder.xml -m_dp models/duration_predictor.xml -m_b models/bert.xml -i mc_menu.json -v vocab/vocab_bert.txt
    [INFO] 2021-04-06 17:27:21,512 Creating Inference Engine
    [INFO] 2021-04-06 17:27:21,513 Loading mel network files:
        models/mel.xml
        models/mel.bin
    [INFO] 2021-04-06 17:27:21,515 Loading melgan network files:
        models/melgan.xml
        models/melgan.bin
    [INFO] 2021-04-06 17:27:21,542 Loading decoder network files:
        models/decoder.xml
        models/decoder.bin
    [INFO] 2021-04-06 17:27:21,593 Loading encoder network files:
        models/encoder.xml
        models/encoder.bin
    [INFO] 2021-04-06 17:27:21,647 Loading duration predictor network files:
        models/duration_predictor.xml
        models/duration_predictor.bin
    [INFO] 2021-04-06 17:27:21,651 Loading bert network files:
        models/bert.xml
        models/bert.bin
    [INFO] 2021-04-06 17:27:21,785 Loading model to the plugin
    [INFO] 2021-04-06 17:27:22,640 Inputs number: 3
        - IteratorGetNext/placeholder_out_port_0 : [1, 256]
        - IteratorGetNext/placeholder_out_port_1 : [1, 256]
        - IteratorGetNext/placeholder_out_port_3 : [1, 256]
    [INFO] 2021-04-06 17:27:22,640 Outputs number: 2
        - unstack/Squeeze_ : [1, 256]
        - unstack/Squeeze_527 : [1, 256]
    **********read_squad_examples complete!**********
    [INFO] 2021-04-06 17:27:22,688 Load 76 examples
    Content:  麦当劳目前的餐点有：大麦克价格为72元、双层牛肉吉事堡价格为62元、嫩煎鸡腿堡价格为82元、麦香鸡价格为44元、麦克鸡块(6块)价格为60元、麦克鸡块(10块)价格为100元、劲辣鸡腿堡价格为72元、麦脆鸡腿(2块)价格为110元、麦脆鸡翅(2块)价格为90元、黄金起司猪排堡价格为52元、麦香鱼价格为44元、烟熏鸡肉长堡价格为74元、姜烧猪肉长堡价格为74元、BLT安格斯黑牛堡价格为109元、BLT辣脆鸡腿堡价格为109元、BLT嫩煎鸡腿堡价格为109元、蕈菇安格斯黑牛堡价格为119元、凯萨脆鸡沙拉价格为99元和义式烤鸡沙拉价格为99元。
    Question:  大麦克多少钱？
    Answer:  72元
    七十二元
    processing 七十二元
    Building prefix dict from the default dictionary ...
    [DEBUG] 2021-04-06 17:27:22,934 Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    [DEBUG] 2021-04-06 17:27:22,934 Loading model from cache /tmp/jieba.cache
    Loading model cost 0.447 seconds.
    [DEBUG] 2021-04-06 17:27:23,381 Loading model cost 0.447 seconds.
    Prefix dict has been built successfully.
    [DEBUG] 2021-04-06 17:27:23,381 Prefix dict has been built successfully.
    /usr/lib/python3/dist-packages/apport/report.py:13: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
    import fnmatch, glob, traceback, errno, sys, atexit, locale, imp
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    ValueError: Unknown Blob precision: BOOL
    Exception ignored in: 'openvino.inference_engine.ie_api.BlobBuffer._get_blob_format'
    ValueError: Unknown Blob precision: BOOL
    [INFO] 2021-04-06 17:27:24,319 Generated tts.wav.
    [INFO] 2021-04-06 17:27:24,319 This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

    ```

    > Once the process is done, we can find the `tts.wav` in the working directory.
