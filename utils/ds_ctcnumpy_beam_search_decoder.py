#
# John Feng(john.feng@intel.com)
# SPDX-License-Identifier: Mozilla Public License 2.0
#
import numpy as np
import ds_ctcdecoder

from multiprocessing import cpu_count
from scipy.special import softmax


class DSCtcnumpyBeamSearchDecoder:
    def __init__(self, alphabet_type, beam_size, max_candidates=None, cutoff_prob=1.0, cutoff_top_n=40,
            scorer_lm_fname=None, alpha=0.75, beta=1.85):
        print (alphabet_type )
        if alphabet_type == 'utf-8':
            self.alphabet = ds_ctcdecoder.UTF8Alphabet()
        # else:
        #     alphabet = ds_ctcdecoder.Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

        try:
            self.num_processes = cpu_count()
        except NotImplementedError:
            self.num_processes = 1
        self.scorer = None
        if scorer_lm_fname is not None:
            self.scorer = ds_ctcdecoder.Scorer(alpha=alpha, beta=beta, 
                scorer_path=scorer_lm_fname, alphabet=self.alphabet)

        self.beam_size = beam_size
        # self.max_candidates = max_candidates
        self.max_candidates = 10
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n

    def decode(self, probs):
        if len(probs.shape) is 2:
            probs = np.expand_dims(probs, axis=0)
        
        ds_beam_results = ds_ctcdecoder.ctc_beam_search_decoder_batch(probs, [np.uint32(prob.shape[0]) for prob in probs], self.alphabet, self.beam_size,
                            self.num_processes, cutoff_prob=self.cutoff_prob, cutoff_top_n=self.cutoff_top_n, 
                            scorer=self.scorer, num_results=self.max_candidates)
        beam_results = [
            dict(
                conf=ds_beam_result[0],
                text=ds_beam_result[1],
            )
            for ds_beam_result in ds_beam_results[0] 
        ]
        return beam_results
