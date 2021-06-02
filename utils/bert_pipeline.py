import os.path

import numpy as np
from openvino.inference_engine import IECore

import utils.tokenization_utils as tokenization_utils


class BertSquadPipeline:
    def __init__(self, vocab, model, model_bin=None, max_seq_length=None, doc_stride=None,
            max_query_length=None, max_answer_length=None, num_of_best_set=None, ie=None,
            device='CPU', ie_extensions=[]):
        """
            Args:
        vocab (str), filename of vocabulary file for bert model
        model (str), filename of IE IR .xml file of the network
        model_bin (str), filename of IE IR .xml file of the network (default (None) is the same as :model:, but
            with extension replaced with .bin)
        max_seq_length (int), the maximum total input sequence length after WordPiece tokenization
        doc_stride (int), how much stride to take between chunks when splitting up a long document into chunks 
        max_query_length (int), the maximum number of tokens for the question and
            questions longer than this will be truncated to this length
        max_answer_length (int), the maximum length of an answer that can be generated
        num_of_best_set (int), the number for n-best predictions to generate the final result
        ie (IECore or None), IECore object to run NN inference with.  Default is to use ie_core_singleton module.
            (default None)
        device (str), inference device for IE, passed here to 1. set default device, and 2. check supported node types
            in the model load; None = do not check (default 'CPU')
        ie_extensions (list(tuple(str,str))), list of IE extensions to load, each extension is defined by a pair
            (device, filename). Records with filename=None are ignored.  (default [])
        """
        # model parameters
        # self.num_batch_frames = 16

        self.vocab_file = vocab
        self.max_seq_length = max_seq_length
        if self.max_seq_length is None:
            self.max_seq_length = 256

        self.doc_stride = doc_stride
        if self.doc_stride is None:
            self.doc_stride = 128

        self.max_query_length = max_query_length
        if self.max_query_length is None:
            self.max_query_length = 64

        self.max_answer_length = max_answer_length
        if self.max_answer_length is None:
            self.max_answer_length = 30

        self.num_of_best_set = num_of_best_set
        if self.num_of_best_set is None:
            self.num_of_best_set = 10

        self.net = self.exec_net = None
        self.default_device = device

        self.input_names = self.output_names = None

        self.ie = ie if ie is not None else IECore()
        self._load_net(model, model_bin_fname=model_bin, device=device, ie_extensions=ie_extensions)

        if device is not None:
            self.activate_model(device)

    def _load_net(self, model_xml_fname, model_bin_fname=None, ie_extensions=[], device='CPU', device_config=None):
        """
        Load IE IR of the network,  and optionally check it for supported node types by the target device.
        model_xml_fname (str)
        model_bin_fname (str or None)
        ie_extensions (list of tuple(str,str)), list of plugins to load, each element is a pair
            (device_name, plugin_filename) (default [])
        device (str or None), check supported node types with this device; None = do not check (default 'CPU')
        device_config
        """
        if model_bin_fname is None:
            model_bin_fname = os.path.basename(model_xml_fname).rsplit('.', 1)[0] + '.bin'
            model_bin_fname = os.path.join(os.path.dirname(model_xml_fname), model_bin_fname)

        # Plugin initialization for specified device and load extensions library if specified
        for extension_device, extension_fname in ie_extensions:
            if extension_fname is None:
                continue
            self.ie.add_extension(extension_path=extension_fname, device_name=extension_device)

        # Read IR
        self.net = self.ie.read_network(model=model_xml_fname, weights=model_bin_fname)

        if self.input_names is None:
            self.input_names = list(self.net.input_info.keys())
        if self.output_names is None:
            self.output_names = list(self.net.outputs.keys())

    def activate_model(self, device):
        if self.exec_net is not None:
            return  # Assuming self.net didn't change
        # Loading model to the plugin
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

    def paragraph_reader(self, paragraph_file):
        return open(paragraph_file, "r").read()[:-1]

    def export_feature_from_file(self, data_file):
        examples, features = tokenization_utils.export_feature(
            vocab_file = self.vocab_file,
            data_file = data_file,
            do_lower_case = False, 
            max_seq_length = self.max_seq_length,
            doc_stride = self.doc_stride,
            max_query_length = self.max_query_length
        )

        return examples, features

    def export_feature_from_text(self, paragraph_text, question, do_lower_case=False):
        examples, features = tokenization_utils.export_feature_from_text(
            vocab_file = self.vocab_file,
            paragraph_text = paragraph_text,
            question_text = question,
            do_lower_case = False, 
            max_seq_length = self.max_seq_length,
            doc_stride = self.doc_stride,
            max_query_length = self.max_query_length
        )
        return examples, features

    def run_squad(self, examples, features):
        assert self.exec_net is not None, "Need to call mds.activate(device) method before mds.stt(...)"

        infer_feature = []
        for _, _ftr in enumerate(features):
            infer_feature.append(_ftr)

        n_best_results = []

        for i, feature in enumerate(infer_feature):
            inputs = {
                self.input_names[0]: np.array([feature.input_ids], dtype=np.int32),
                self.input_names[1]: np.array([feature.input_mask], dtype=np.int32),
                self.input_names[2]: np.array([feature.segment_ids], dtype=np.int32),
            }

            res = self.exec_net.infer(inputs=inputs)

            start_logits = res[self.output_names[0]].flatten()
            end_logits = res[self.output_names[1]].flatten()

            start_logits = start_logits - np.log(np.sum(np.exp(start_logits)))
            end_logits = end_logits - np.log(np.sum(np.exp(end_logits)))

            sorted_start_index = np.argsort(-start_logits)
            sorted_end_index = np.argsort(-end_logits)

            token_length = len(feature.tokens)

            for _s_idx in sorted_start_index[:self.num_of_best_set]:
                for _e_idx in sorted_end_index[:self.num_of_best_set]:
                    if _s_idx >= token_length:
                        continue
                    if _e_idx >= len(feature.tokens):
                        continue
                    if _s_idx not in feature.token_to_orig_map:
                        continue
                    if _e_idx not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(_s_idx, False):
                        continue
                    if _e_idx < _s_idx:
                        continue
                    length = _e_idx - _s_idx + 1
                    if length > self.max_answer_length:
                        continue
                    n_best_results.append((start_logits[_s_idx] +  end_logits[_e_idx], 
                        "".join(examples[0].doc_tokens[feature.token_to_orig_map[_s_idx]:feature.token_to_orig_map[_e_idx] + 1])))

        max_prob = -100000
        best_result = ""
        for _res in n_best_results:
            _prob, _text = _res
            if _prob > max_prob:
                max_prob = _prob
                best_result = _text

        return best_result