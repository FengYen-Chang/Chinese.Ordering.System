#!/usr/bin/env python
"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
import wave

from argparse import ArgumentParser, SUPPRESS
import yaml
import numpy as np
from tqdm import tqdm
import logging as log
formatter = '[%(levelname)s] %(asctime)s %(message)s'
log.basicConfig(level=log.INFO, format=formatter)
import pycnnum

from utils.deep_speech_pipeline import DeepSpeechPipeline, PROFILES
from utils.bert_pipeline import BertSquadPipeline
from utils.text_to_speech_pipeline import Text2SpeechPipeline

from openvino.inference_engine import IECore
import scipy.io.wavfile as wavfile


def build_argparser():
    parser = ArgumentParser(add_help=False, description="Ordering System demo")
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m_mel", "--model-mel", help="Required. Path to an .xml file with a trained mel model.", required=True,
                      type=str)
    args.add_argument("-m_mg", "--model-melgan", help="Required. Path to an .xml file with a trained mel gan model.", required=True,
                      type=str)
    args.add_argument("-m_d", "--model-decoder", help="Required. Path to an .xml file with a trained decoder model.", required=True,
                      type=str)
    args.add_argument("-m_e", "--model-encoder", help="Required. Path to an .xml file with a trained encoder model.", required=True,
                      type=str)
    args.add_argument("-m_dp", "--model-duration_predictor", help="Required. Path to an .xml file with a trained duration predictor model.", required=True,
                      type=str)
    args.add_argument("-m_b", "--model-bert", help="Required. Path to an .xml file with a trained bert model.", required=True,
                      type=str)
    args.add_argument("-m_a", "--model-audio", help="Required. Path to an .xml file with a trained audio model.", required=True,
                      type=str)
    args.add_argument('-L', '--lm', type=str, metavar="FILENAME",
                      help="path to language model file (optional)")
    args.add_argument("-i", "--input", help="Required. Path to a 16k .wav file for audio model.",
                      required=True,
                      type=str)#, nargs="+")
    args.add_argument('-p', '--profile', type=str, metavar="NAME", required=True,
                      help="Choose pre/post-processing profile: "
                           "mds06x_en for Mozilla DeepSpeech v0.6.x, "
                           "mds07x_en or mds08x_en for Mozilla DeepSpeech v0.7.x/x0.8.x, "
                           "mds09x_cn for Mozilla DeepSpeech v0.9.x Chinese Model, "
                           "other: filename of a YAML file (required)")
    args.add_argument("-para", "--paragraph", help="Required. Path to a .txt file w/ paragraph for bert.",
                      required=True,
                      type=str)
    args.add_argument("-v_b", "--vocab-bert", help="Required. Path to vocabulary file for bert model.", required=True, type=str)
    args.add_argument("-v_p", "--vocab-pinyin", help="Required. Path to pinyin vocabulary file for TTS model.", required=True, type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    
    args.add_argument('-b', '--beam-width', type=int, default=500, metavar="N",
                      help="Beam width for beam search in CTC decoder (default 500)")
    args.add_argument('-c', '--max-candidates', type=int, default=1, metavar="N",
                      help="Show top N (or less) candidates (default 1)")
    args.add_argument("--max-seq-length", 
                      help="Optional. The maximum total input sequence length after WordPiece tokenization. "
                      , default=256, type=int)
    args.add_argument("--doc-stride", 
                      help="Optional.When splitting up a long document into chunks, how much stride to "
                      "take between chunks.", default=128, type=int)
    args.add_argument("--max-query-length", 
                      help="Optional. The maximum number of tokens for the question. Questions longer than "
                      "this will be truncated to this length.", default=64, type=int)
    args.add_argument("-mal", "--max_answer_length", 
                      help="Optional. The maximum length of an answer that can be generated.", default=30, type=int)
    args.add_argument("-nbest", "--num_of_best_set", 
                      help="Optional. The number for n-best predictions to generate the final result", default=10, type=int)

    args.add_argument('-l', '--cpu_extension', type=str, metavar="FILENAME",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.")

    return parser

def get_profile(profile_name):
    if profile_name in PROFILES:
        return PROFILES[profile_name]
    with open(profile_name, 'rt') as f:
        profile = yaml.safe_load(f)
    return profile

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # STT
    profile = get_profile(args.profile)

    stt = DeepSpeechPipeline(
        model = args.model_audio,
        lm = args.lm,
        beam_width = args.beam_width,
        max_candidates = args.max_candidates,
        profile = profile,
        device = args.device,
        ie_extensions = [(args.device, args.cpu_extension)] if args.device == 'CPU' else [],
    )

    # BERT - SQUAD
    squad = BertSquadPipeline(
        vocab = args.vocab_bert,
        model = args.model_bert,
        max_seq_length = args.max_seq_length,
        doc_stride = args.doc_stride,
        max_query_length = args.max_query_length,
        max_answer_length = args.max_answer_length,
        num_of_best_set = args.num_of_best_set,
        device = args.device,
        ie_extensions = [(args.device, args.cpu_extension)] if args.device == 'CPU' else [],
    )

    # TTS
    tts = Text2SpeechPipeline(
        vocab = args.vocab_pinyin,
        model_mel = args.model_mel,
        model_melgan = args.model_melgan,
        model_decoder = args.model_decoder,
        model_encoder = args.model_encoder,
        model_duration_predictor = args.model_duration_predictor,
        device = args.device,
        ie_extensions = [(args.device, args.cpu_extension)] if args.device == 'CPU' else [],
    )

    # Run STT
    wave_read = wave.open(args.input, 'rb')
    channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
    assert sample_width == 2, "Only 16-bit WAV PCM supported"
    assert compression_type == 'NONE', "Only linear PCM WAV files supported"
    assert channel_num == 1, "Only mono WAV PCM supported"
    audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((pcm_length, channel_num))
    wave_read.close()

    audio_features = stt.extract_mfcc(audio, sampling_rate=sampling_rate)
    character_probs = stt.extract_per_frame_probs(audio_features, wrap_iterator=tqdm)
    transcription = stt.decode_probs(character_probs)

    # Run BERT - SQUAD
    paragraph_text = squad.paragraph_reader(args.paragraph)

    examples, features = squad.export_feature_from_text(
                            paragraph_text = paragraph_text,
                            question = transcription[0]['text'],
                        )

    print ("Content: ", "".join(examples[0].doc_tokens))
    print ("Question: ", examples[0].question_text)
    
    bert_sentence = squad.run_squad(examples, features)

    # Run TTS
    sentence = pycnnum.num2cn(int(bert_sentence[:-1])) + bert_sentence[-1]
    cn_sentence_seq, py_sentence_seq = tts.process_tts_input(sentence)

    generated_mel = tts.run_synthesize(py_sentence_seq, cn_sentence_seq)

    generated_mel *= 32767 / max(0.01, np.max(np.abs(generated_mel)))
    wavfile.write('tts.wav', 22050, generated_mel.astype(np.int16))
    log.info("Generated tts.wav.")
    
    log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")

if __name__ == '__main__':
    sys.exit(main() or 0)
