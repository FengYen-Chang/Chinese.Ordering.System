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
from argparse import ArgumentParser, SUPPRESS
import numpy as np
import logging as log

import utils

from openvino.inference_engine import IECore
import scipy.io.wavfile as wavfile


def build_argparser():
    parser = ArgumentParser(add_help=False, description="TTS demo")
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m_mel", "--model_mel", help="Required. Path to an .xml file with a trained mel model.", required=True,
                      type=str)
    args.add_argument("-m_mg", "--model_melgan", help="Required. Path to an .xml file with a trained mel gan model.", required=True,
                      type=str)
    args.add_argument("-m_d", "--model_decoder", help="Required. Path to an .xml file with a trained decoder model.", required=True,
                      type=str)
    args.add_argument("-m_e", "--model_encoder", help="Required. Path to an .xml file with a trained encoder model.", required=True,
                      type=str)
    args.add_argument("-m_dp", "--model_duration_predictor", help="Required. Path to an .xml file with a trained duration predictor model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a text which you want to make as speech.",
                      required=True,
                      type=str)#, nargs="+")
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)

    return parser

def preprocess(phone, py2idx):
    sequence = np.array([py2idx[p] for p in phone.split()])
    sequence = np.stack([sequence])

    return sequence.astype(np.int64)

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = lengths.max()
    # ids = np.arange(0, max_len).unsqueeze(
    #     0).expand(batch_size, -1)
    ids = np.broadcast_to(np.expand_dims(np.arange(0, max_len), 0), (batch_size, max_len))
    # mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    mask = np.broadcast_to(ids >= np.expand_dims(lengths, 1), (batch_size, max_len))

    return mask

def synthesize(decoder, encoder, duration_predictor, mel,
               melgan, 
               py_text_seq, cn_text_seq, 
            #    variance_adaptor=variance_adaptor, 
               real_len=None, duration_control=1.0,prefix=''):
    def expand(batch, predicted, max_length=200):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            print (expand_size)
            # out.append(vec.expand(int(expand_size), -1))
            out.append(np.broadcast_to(vec, (int(expand_size), 256)))
        out = np.concatenate(out, 0)
        ori_len = out.shape[0]
        out = np.concatenate((out, np.zeros((max_length - out.shape[0] % max_length, out.shape[1]), dtype=np.float32)), 0)
        return out, ori_len
    
    def LR(x, duration, max_len=None):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded, ori_len = expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        return output, np.array(mel_len, dtype=np.int64), ori_len

    duration_mean = 18.877746355061273
    mel_mean = -6.0304103

    max_input_len = 10

    # fill inputs
    cn_text_seq = np.pad(cn_text_seq, ((0, 0), (0, (max_input_len - real_len % max_input_len))), 
                    mode='constant', constant_values=cn_text_seq[0][0])
    py_text_seq = np.pad(py_text_seq, ((0, 0), (0, (max_input_len - real_len % max_input_len))), 
                    mode='constant', constant_values=py_text_seq[0][0])

    src_len = np.array([py_text_seq.shape[1]]).astype(np.int64)
    src_mask = get_mask_from_lengths(src_len)

    encoder_output_l = []
    dp_output_l = []
    decoder_output_l = []
    mel_output_l = []

    for i in range(0, py_text_seq.shape[1], max_input_len):
        res_e = encoder.infer(inputs={'hz_seq': cn_text_seq[0][i:i+max_input_len],
                                  'src_mask': src_mask[0][i:i+max_input_len], 
                                  'src_seq': py_text_seq[0][i:i+max_input_len]})

        encoder_output_l.append(res_e['encoder_output'])
    encoder_output = np.concatenate(encoder_output_l, axis=1)

    for i in range(len(encoder_output_l)):
        res_dp = duration_predictor.infer(inputs={'encoder_output': encoder_output_l[i],
                                              'src_mask': src_mask[0][i:i+max_input_len]})
        dp_output_l.append(res_dp['duration_predictor_output'])
    dp_output = np.concatenate(dp_output_l, axis=1)

    if real_len:
        dp_output_r = dp_output[:, :real_len]
        encoder_output_r = encoder_output[:, :real_len, :]

    d_rounded = np.clip(np.round((dp_output_r + duration_mean) * duration_control),
                        a_min=0.0, a_max=None)
    va_output, mel_len, ori_len = LR(encoder_output_r, d_rounded)
    mel_mask = get_mask_from_lengths(mel_len)


    for i in range(0, mel_mask.shape[1], 200):
        res_decoder = decoder.infer(inputs={'mel_mask': mel_mask[0][i:i+200],
                                        'variance_adaptor_output': va_output[0][i:i+200]})
        decoder_output_l.append(res_decoder['decoder_output'])

    decoder_output = np.concatenate(decoder_output_l, axis=1)
    
    for i in range(0, decoder_output.shape[1], 200):
        res_mel = mel.infer(inputs={"decoder_output": decoder_output[0][i:i+200]})
        mel_output_l.append(res_mel['mel_output'])

    mel_output = np.concatenate(mel_output_l, axis=1)
    mel_output = np.transpose(mel_output + mel_mean, (0, 2, 1))

    melgan_output_l = []
    for i in range(0, mel_output.shape[2], 200):
        res_melgan = melgan.infer(inputs={'0': mel_output[:, :, i:i+200]})
        melgan_output_l.append(res_melgan['Tanh_101'])
    melgan_output = np.concatenate(melgan_output_l, axis=2)
  
    return melgan_output[:, :, :ori_len * 256]

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_mel_xml = args.model_mel
    model_mel_bin = os.path.splitext(model_mel_xml)[0] + ".bin"

    model_mg_xml = args.model_melgan
    model_mg_bin = os.path.splitext(model_mg_xml)[0] + ".bin"

    model_decoder_xml = args.model_decoder
    model_decoder_bin = os.path.splitext(model_decoder_xml)[0] + ".bin"

    model_encoder_xml = args.model_encoder
    model_encoder_bin = os.path.splitext(model_encoder_xml)[0] + ".bin"

    model_dp_xml = args.model_duration_predictor
    model_dp_bin = os.path.splitext(model_dp_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()

    # Read IR
    log.info("Loading mel network files:\n\t{}\n\t{}".format(model_mel_xml, model_mel_bin))
    mel_net = ie.read_network(model=model_mel_xml, weights=model_mel_bin)

    log.info("Loading melgan network files:\n\t{}\n\t{}".format(model_mg_xml, model_mg_bin))
    mg_net = ie.read_network(model=model_mg_xml, weights=model_mg_bin)

    log.info("Loading decoder network files:\n\t{}\n\t{}".format(model_decoder_xml, model_decoder_bin))
    decoder_net = ie.read_network(model=model_decoder_xml, weights=model_decoder_bin)

    log.info("Loading encoder network files:\n\t{}\n\t{}".format(model_encoder_xml, model_encoder_bin))
    encoder_net = ie.read_network(model=model_encoder_xml, weights=model_encoder_bin)

    log.info("Loading duration predictor network files:\n\t{}\n\t{}".format(model_dp_xml, model_dp_bin))
    dp_net = ie.read_network(model=model_dp_xml, weights=model_dp_bin)

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_mel_net = ie.load_network(network=mel_net, device_name=args.device)
    exec_mg_net = ie.load_network(network=mg_net, device_name=args.device)
    exec_decoder_net = ie.load_network(network=decoder_net, device_name=args.device)
    exec_encoder_net = ie.load_network(network=encoder_net, device_name=args.device)
    exec_dp_net = ie.load_network(network=dp_net, device_name=args.device)

    # Read id_sequence
    with open(os.path.join('./vocab/','vocab_pinyin.txt')) as F:
        py_vocab = F.read().split('\n')
        py_vocab_size = len(py_vocab) 
        py2idx = dict([(c,i) for i,c in enumerate(py_vocab)])

    sentence = args.input
    cn_sentence, _ = utils.split2sent(sentence)

    max_input_len = 10

    print('processing',cn_sentence[0])
    py_sentence = utils.convert(cn_sentence[0])
    py_sentence_seq = preprocess(py_sentence, py2idx)
    
    cn_sentence_seq = utils.convert_cn(cn_sentence[0]).astype(np.int64)

    real_len = cn_sentence_seq.shape[1]

    """
    # fill inputs
    cn_sentence_seq = np.pad(cn_sentence_seq, ((0, 0), (0, max_input_len - real_len)), 
                             mode='constant', constant_values=cn_sentence_seq[0][0])
    py_sentence_seq = np.pad(py_sentence_seq, ((0, 0), (0, max_input_len - real_len)), 
                             mode='constant', constant_values=py_sentence_seq[0][0])
    """

    # Start sync inference
    generated_mel = synthesize(exec_decoder_net, exec_encoder_net, exec_dp_net, exec_mel_net,
                               exec_mg_net, py_sentence_seq, cn_sentence_seq, real_len=real_len)

    generated_mel *= 32767 / max(0.01, np.max(np.abs(generated_mel)))
    wavfile.write('tts.wav', 22050, generated_mel.astype(np.int16))
    log.info("Generated tts.wav.")
    
    log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")

if __name__ == '__main__':
    sys.exit(main() or 0)
