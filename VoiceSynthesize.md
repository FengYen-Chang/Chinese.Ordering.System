# Voice Synthesize System

In an **Ordering System**, the voice synthesize function is a key function to make the reply as voice to improve the interaction, not just some content which show on screen. For the function, we integrate the **FastSpeech2** and **MelGAN** to genarte the mel-spectrogram and raw audio sample, and then make it as `.wav` file for voicing.

In this page, it shows the all steps to enable the Chinese base FastSpeech2 and MelGAN algorithm by OpenVINO. 

### Preparation

* Download FastSpeech2 and MelGAN submodules

    ```sh
    git submodule update --init ./extension/mandarin-tts
    git submodule update --init ./extension/melgan 
    ```

* Please check the [Synthesis(Inference)](./extension/mandarin-tts/README.md#synthesis-inference) section in readme file and download the pre-trained FastSpeech2 model from goolge drive

### Prerequisites

Please check the [Dependencies](./extension/mandarin-tts/README.md#dependencies) section in readme file of **mandarin-tts** and [Prerequisites](./extension/melgan/README.md#prerequisites) section in **melgan**.

> To enable both model, we recommand to use docker or virtual environment for each model to keep the environment clean.

### Export ONNX Model

Before convert the model to Intermediate Representation (IR) for OpenVINO, we need to convert the PyTorch model 

* FastSpeech2

    As the length of context is different, therefore, we fix the input size as `10`, it means the converted model can accept **8** Chinese characters with 2 token for start and end, to make it generate a static model.

    ```sh
    cd extension/mandarin-tts
    mkdir onnx

    export PATH_TO_CKPT=/path/to/download/checkpoint
    export OUTPUT_DIR=/path/to/output/folder
    export INPUT_DIR=/path/to/input/data

    python synthesize.py \
        --model_file ${PATH_TO_CKPT}/checkpoint_300000.pth.tar \
        --text_file ${INPUT_DIR}/test.txt \ 
        --channel 2 \
        --duration_control 1.0 \
        --output_dir ${OUTPUT_DIR}
    ```

    > For the `test.txt`, you can input any content as Chinese characters and the length is lower than `8`. For example, `七十二元`.

    After execute the `synthesize.py`, you could see the `*.onnx` models under directory, `onnx`.

    ```sh
    ls onnx -al

    -rw-r--r--  1 root root 47241615 Mar 16 05:29 decoder.onnx
    -rw-r--r--  1 root root  1582451 Mar 16 05:29 duration_predictor.onnx
    -rw-r--r--  1 root root 53126378 Mar 16 05:29 encoder.onnx
    -rw-r--r--  1 root root    82468 Mar 16 05:29 mel.onnx
    ```

* MelGAN

    Same as FastSpeech2, we also assign the fix size for MelGAN to make it generate a static model.

    ```sh
    cd extension/melgan
    mkdir onnx

    python export_onnx.py
    ```

    After execute the `export_onnx.py`, you could see the `melgan.onnx` models under current directory.

    ```sh
    ls -al | grep onnx

    -rw-r--r-- 1 root root      891  三  10 11:00 export_onnx.py
    -rw-r--r-- 1 root root 17068676  三  10 11:00 melgan.onnx
    ```

### Convert the all .onnx model to Intermediate Representation (IR)

Please use below command to run the Model Optimizer for those converted `.onnx` model to get the IR model

```sh
export MODEL_DIR=/path/to/converted/onnx/model

python mo.py --input_model=${MODEL_DIR}/${CONVERTED_ONNX_MODEL}
```
> Please convert those `.onnx` model one-by-one.


