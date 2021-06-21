import os.path

import numpy as np
from openvino.inference_engine import IECore

import utils.hz_utils as hz_utils
import utils.utils as utils


class Text2SpeechPipeline:
    def __init__(self, vocab, model_mel, model_melgan, model_decoder, model_encoder, model_duration_predictor,
            model_mel_bin=None, model_melgan_bin=None, model_decoder_bin=None, model_encoder_bin=None, model_duration_predictor_bin=None,
            max_input_len=10, ie=None, device='CPU', ie_extensions=[]):
        """
            Args:
        vocab (str), filename of vocabulary file for bert model
        model_mel (str), filename of IE IR .xml file of the mel network
        model_mel_bin (str), filename of IE IR .xml file of the mel network (default (None) is the same as :model:, but
            with extension replaced with .bin)
        model_melgan (str), filename of IE IR .xml file of the melgan network
        model_melgan_bin (str), filename of IE IR .xml file of the melgan network (default (None) is the same as :model:, but
            with extension replaced with .bin)
        model_decoder (str), filename of IE IR .xml file of the decoder network
        model_decoder_bin (str), filename of IE IR .xml file of the decoder network (default (None) is the same as :model:, but
            with extension replaced with .bin)
        model_encoder (str), filename of IE IR .xml file of the encoder network
        model_encoder_bin (str), filename of IE IR .xml file of the encoder network (default (None) is the same as :model:, but
            with extension replaced with .bin)
        model_duration_predictor (str), filename of IE IR .xml file of the duration predictor network
        model_duration_predictor_bin (str), filename of IE IR .xml file of the duration predictor network (default (None) is the 
            same as :model:, but with extension replaced with .bin)

            Terminology:
        d, decoder
        e, encoder
        dp, duration predictor
        """
        # model parameters
        # self.num_batch_frames = 16

        self.vocab_file = vocab
        self.pinyin = None
        self.max_input_len = max_input_len
        self.real_len = None

        self.net_mel = self.exec_net_mel = None
        self.net_melgan = self.exec_net_melgan = None
        self.net_d = self.exec_net_d = None
        self.net_e = self.exec_net_e = None
        self.net_dp = self.exec_net_dp = None
        self.default_device = device

        self.net_list = []
        # [self.net_mel, self.net_melgan, self.net_d, self.net_e, self.net_dp]
        self.exec_net_list = []
        # [self.exec_net_mel, self.exec_net_melgan, self.exec_net_d, self.exec_net_e, self.exec_net_dp]

        self.duration_mean = 18.877746355061273
        self.mel_mean = -6.0304103

        if self.pinyin is None:
            self.pinyin = self.get_pinyin2idx(self.vocab_file)

        self.ie = ie if ie is not None else IECore()
        model_list = [model_mel, model_melgan, model_decoder, model_encoder, model_duration_predictor]
        self._load_net(model_xml_fname_list=model_list, model_bin_fname_list=[], device=device, ie_extensions=ie_extensions)

        self.net_mel, self.net_melgan, self.net_d, \
            self.net_e, self.net_dp = self.net_list[0], self.net_list[1], self.net_list[2], self.net_list[3], self.net_list[4]

        if device is not None:
            self.activate_model(device)
            self.exec_net_mel, self.exec_net_melgan, self.exec_net_d, self.exec_net_e, \
                self.exec_net_dp = self.exec_net_list[0], self.exec_net_list[1], self.exec_net_list[2], self.exec_net_list[3], self.exec_net_list[4]

    def _load_net(self, model_xml_fname_list, model_bin_fname_list=None, ie_extensions=[], device='CPU', device_config=None):
        """
        Load IE IR of the network,  and optionally check it for supported node types by the target device.
        model_xml_fname (str)
        model_bin_fname (str or None)
        ie_extensions (list of tuple(str,str)), list of plugins to load, each element is a pair
            (device_name, plugin_filename) (default [])
        device (str or None), check supported node types with this device; None = do not check (default 'CPU')
        device_config
        """
        if len(model_bin_fname_list) is 0:
            for model_xml_fname in model_xml_fname_list:
                model_bin_fname = os.path.basename(model_xml_fname).rsplit('.', 1)[0] + '.bin'
                model_bin_fname = os.path.join(os.path.dirname(model_xml_fname), model_bin_fname)
                model_bin_fname_list.append(model_bin_fname)

        # Plugin initialization for specified device and load extensions library if specified
        for extension_device, extension_fname in ie_extensions:
            if extension_fname is None:
                continue
            self.ie.add_extension(extension_path=extension_fname, device_name=extension_device)

        # Read IR
        for idx, model_xml_fname in enumerate(model_xml_fname_list):
            self.net_list.append(self.ie.read_network(model=model_xml_fname, weights=model_bin_fname_list[idx]))

    def activate_model(self, device):
        if len(self.exec_net_list) is not 0:
            return  # Assuming self.net didn't change
        # Loading model to the plugin
        for _, net in enumerate(self.net_list):
            self.exec_net_list.append(self.ie.load_network(network=net, device_name=device))

    def get_pinyin2idx(self, vocab_file):
        with open(vocab_file) as F:
            py_vocab = F.read().split('\n')
            py2idx = dict([(c,i) for i,c in enumerate(py_vocab)])
        return py2idx

    def process_tts_input(self, sentence):
        cn_sentence, _ = hz_utils.split2sent(sentence)
        py_sentence = hz_utils.convert(cn_sentence[0])
        py_sentence_seq = self.preprocess(py_sentence, self.pinyin)
        cn_sentence_seq = hz_utils.convert_cn(cn_sentence[0]).astype(np.int64)

        self.real_len = cn_sentence_seq.shape[1]
        cn_sentence_seq = np.pad(cn_sentence_seq, 
            ((0, 0), (0, self.max_input_len - self.real_len % self.max_input_len)),
            mode='constant', constant_values=cn_sentence_seq[0][0]
        )
        py_sentence_seq = np.pad(py_sentence_seq, 
            ((0, 0), (0, self.max_input_len - self.real_len % self.max_input_len)),
            mode='constant', constant_values=py_sentence_seq[0][0]
        )

        return cn_sentence_seq, py_sentence_seq

    def _expand(self, batch, predicted, max_length=200):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            # out.append(vec.expand(int(expand_size), -1))
            out.append(np.broadcast_to(vec, (int(expand_size), 256)))
        out = np.concatenate(out, 0)
        ori_len = out.shape[0]
        out = np.concatenate((out, np.zeros((max_length - out.shape[0] % max_length, out.shape[1]), dtype=np.float32)), 0)

        return out, ori_len

    def _LR(self, x, duration, max_len=None):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded, ori_len = self._expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        return output, np.array(mel_len, dtype=np.int64), ori_len

    def preprocess(self, phone, py2idx):
        sequence = np.array([py2idx[p] for p in phone.split()])
        sequence = np.stack([sequence])

        return sequence.astype(np.int64)

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = lengths.max()
        ids = np.broadcast_to(np.expand_dims(np.arange(0, max_len), 0), (batch_size, max_len))
        mask = np.broadcast_to(ids >= np.expand_dims(lengths, 1), (batch_size, max_len))

        return mask

    def run_synthesize(self, py_text_seq, cn_text_seq, duration_control=1.0, prefix=''):
        encoder_output_l = []
        dp_output_l = []
        mel_output_l = []
        melgan_output_l = []

        src_len = np.array([py_text_seq.shape[1]]).astype(np.int64)
        src_mask = self.get_mask_from_lengths(src_len)

        for i in range(0, py_text_seq.shape[1], self.max_input_len):
            res_e = self.exec_net_e.infer(
                        inputs={
                            'hz_seq': cn_text_seq[0][i:i+self.max_input_len],
                            'src_mask': src_mask[0][i:i+self.max_input_len], 
                            'src_seq': py_text_seq[0][i:i+self.max_input_len]
                        })

            res_dp = self.exec_net_dp.infer(
                        inputs={
                            'encoder_output': res_e['encoder_output'],
                            'src_mask': src_mask[0][i:i+self.max_input_len]
                        })

            encoder_output_l.append(res_e['encoder_output'])
            dp_output_l.append(res_dp['duration_predictor_output'])

        encoder_output = np.concatenate(encoder_output_l, axis=1)
        dp_output = np.concatenate(dp_output_l, axis=1)

        if self.real_len:
            encoder_output_r, dp_output_r = encoder_output[:, :self.real_len, :], dp_output[:, :self.real_len]

        d_rounded = np.clip(
                        np.round((dp_output_r + self.duration_mean) * duration_control),
                            a_min=0.0, a_max=None)
        va_output, mel_len, ori_len = self._LR(encoder_output_r, d_rounded)
        mel_mask = self.get_mask_from_lengths(mel_len)

        for i in range(0, mel_mask.shape[1], 200):
            res_decoder = self.exec_net_d.infer(
                            inputs={
                                'mel_mask': mel_mask[0][i:i+200],            
                                'variance_adaptor_output': va_output[0][i:i+200]
                            })
            res_mel = self.exec_net_mel.infer(
                            inputs={
                                "decoder_output": res_decoder['decoder_output']
                            })
            
            mel_output_l.append(res_mel['mel_output'])

        mel_output = np.concatenate(mel_output_l, axis=1)
        mel_output = np.transpose(mel_output + self.mel_mean, (0, 2, 1))

        for i in range(0, mel_output.shape[2], 200):
            res_melgan = self.exec_net_melgan.infer(inputs={'0': mel_output[:, :, i:i+200]})
            melgan_output_l.append(res_melgan['Tanh_101'])
        melgan_output = np.concatenate(melgan_output_l, axis=2)
        
        return melgan_output[:, :, :ori_len * 256]
