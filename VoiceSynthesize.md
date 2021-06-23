# Voice Synthesize System

In an **Ordering System**, the voice synthesize function is a key function to make the reply as voice to improve the interaction, not just some content which show on screen. For the function, we integrate the **mandarin-tts** which based on fastspeech2 to genarte the mel-spectrogram and **MelGAN** to genarte the raw audio sample, and then make it as `.wav` file for voicing.

In this page, it shows the all steps to enable the Chinese base FastSpeech2 and MelGAN algorithm by OpenVINO. 

### Preparation

* Download mandarin-tts and MelGAN submodules

    ```sh
    git submodule update --init ./extension/mandarin-tts
    git submodule update --init ./extension/melgan 
    ``` 
    > To enable both model, we recommand to use docker or virtual environment for each model to keep the environment clean.

### mandarin-tts

* Dependencies

    Please run below command to install all dependencies.
    
    ```sh
    cd ./extension/mandarin-tts
    pip install -r requirements.txt
    ```
    
    Download the pre-trained model and extract it.
    
    ```sh
    pip install gdown
    gdown https://drive.google.com/uc?id=11mBus5gn69_KwvNec9Zy9jjTs3LgHdx3
    tar xf fastspeech2u_ckpt.tar.gz 
    ```

* Export ONNX Model

    ```sh
    python export_onnx.py --model_file ./ckpt/hanzi/checkpoint_300000.pth.tar --text_file ./test.txt --channel 2 --duration_control 1.0 --output_dir ./output
    ```

    > After run `export_onnx.py`, you could see the folder `onnx` in current working directory and it includes 4 `.onnx` models, `decoder.onnx`, `duration_predictor.onnx`, `encoder.onnx` and `mel.onnx`.
   
* Convert the all .onnx model to Intermediate Representation (IR)

    Please use below command to run the Model Optimizer for those converted `.onnx` model to IR model.

    ```sh
    export MODEL_DIR=/path/to/converted/onnx/model
    export OUTPUT_DIR=/path/to/converted/IR/model

    python mo.py --input_model=${MODEL_DIR}/decoder.onnx -o ${OUTPUT_DIR}
    python mo.py --input_model=${MODEL_DIR}/duration_predictor.onnx -o ${OUTPUT_DIR}
    python mo.py --input_model=${MODEL_DIR}/encoder.onnx -o ${OUTPUT_DIR}
    python mo.py --input_model=${MODEL_DIR}/mel.onnx -o ${OUTPUT_DIR}
    ```


### MelGAN

* Export ONNX Model

    ```sh
    cd extension/melgan
    python3 export_onnx.py
    ```

    > After run `export_onnx.py`, you could see the folder `onnx` in current working directory and it includes 1 `.onnx` model, `melgan.onnx`.
   
* Convert the all .onnx model to Intermediate Representation (IR)

    Please use below command to run the Model Optimizer for those converted `.onnx` model to IR model.

    ```sh
    export MODEL_DIR=/path/to/converted/onnx/model
    export OUTPUT_DIR=/path/to/converted/IR/model

    python mo.py --input_model=${MODEL_DIR}/melgan.onnx -o ${OUTPUT_DIR}
    ```


