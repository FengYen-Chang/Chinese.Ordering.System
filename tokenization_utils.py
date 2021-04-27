import collections
import logging
import json
import six
import tokenization


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               input_span_mask,
               start_position=None,
               end_position=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.input_span_mask = input_span_mask
    self.start_position = start_position
    self.end_position = end_position

def get_doc_token(paragraph_text, do_lower_case=False):
  raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=do_lower_case)
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True

  k = 0
  temp_word = ""
  for c in paragraph_text:
    if tokenization._is_whitespace(c):
      char_to_word_offset.append(k-1)
      continue
    else:
      temp_word += c
      char_to_word_offset.append(k)
    if do_lower_case:
      temp_word = temp_word.lower()
    if temp_word == raw_doc_tokens[k]:
      doc_tokens.append(temp_word)
      temp_word = ""
      k += 1

  assert k==len(raw_doc_tokens)

  return doc_tokens

#
def customize_tokenizer(text, do_lower_case=False):
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
  temp_x = ""
  text = tokenization.convert_to_unicode(text)
  for c in text:
    if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(c) or tokenization._is_control(c):
      temp_x += " " + c + " "
    else:
      temp_x += c
  if do_lower_case:
    temp_x = temp_x.lower()
  return temp_x.split()

#
class ChineseFullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=False):
    self.vocab = tokenization.load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
    self.do_lower_case = do_lower_case
  def tokenize(self, text):
    split_tokens = []
    for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return tokenization.convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return tokenization.convert_by_vocab(self.inv_vocab, ids)

#
def read_squad_examples(input_file, is_training, do_lower_case=False):
  """Read a SQuAD json file into a list of SquadExample."""
  with open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  #
  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      doc_tokens = get_doc_token(paragraph_text, do_lower_case=do_lower_case)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None

        if is_training:
          answer = qa["answers"][0]
          orig_answer_text = answer["text"]

          if orig_answer_text not in paragraph_text:
            print("Could not find answer")
          else:
            answer_offset = paragraph_text.index(orig_answer_text)
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = "".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = "".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if do_lower_case:
                cleaned_answer_text = cleaned_answer_text.lower()
            if actual_text.find(cleaned_answer_text) == -1:
              pdb.set_trace()
              print(("Could not find answer: '%s' vs. '%s'")% (actual_text, cleaned_answer_text))
              continue

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)
  print("**********read_squad_examples complete!**********")
  
  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  # tokenizer = ChineseFullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  features = []

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      input_span_mask = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      input_span_mask.append(1)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
        input_span_mask.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)
      input_span_mask.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
        input_span_mask.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)
      input_span_mask.append(0)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        input_span_mask.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(input_span_mask) == max_seq_length

      start_position = None
      end_position = None
      if is_training:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      # if example_index < 3:
        # print("*** Example ***")
        # print("unique_id: %s" % (unique_id))
        # print("example_index: %s" % (example_index))
        # print("doc_span_index: %s" % (doc_span_index))
        # print("tokens: %s" % " ".join(
        #    [tokenization.printable_text(x) for x in tokens]))
        # print("token_to_orig_map: %s" % " ".join(
        #     ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        # print("token_is_max_context: %s" % " ".join([
        #     "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        # ]))
        # print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # print(
        #     "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # print(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # print(
        #   "input_span_mask: %s" % " ".join([str(x) for x in input_span_mask]))
        # if is_training:
        #   answer_text = " ".join(tokens[start_position:(end_position + 1)])
        #   print("start_position: %d" % (start_position))
        #   print("end_position: %d" % (end_position))
        #   print(
        #       "answer: %s" % (tokenization.printable_text(answer_text)))


      feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          input_span_mask=input_span_mask,
          start_position=start_position,
          end_position=end_position)
      unique_id += 1

      features.append(feature)

  return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index

def export_feature(vocab_file, data_file, do_lower_case, max_seq_length, doc_stride, max_query_length):
  _is_training = False
  
  tokenizer = ChineseFullTokenizer(
    vocab_file=vocab_file, do_lower_case=False)
  examples = read_squad_examples(
      input_file=data_file, is_training=_is_training)
  logging.info("Load {} examples".format(len(examples)))
  features = convert_examples_to_features(examples, tokenizer, max_seq_length,doc_stride, max_query_length, _is_training)
  return examples, features

def export_feature_from_text(vocab_file, paragraph_text, question_text, 
                             do_lower_case, max_seq_length, doc_stride, max_query_length):
  _is_training = False
  tokenizer = ChineseFullTokenizer(
    vocab_file=vocab_file, do_lower_case=do_lower_case)
  examples = [SquadExample(
            qas_id='runtime_question_for_demo',
            question_text=question_text,
            doc_tokens=get_doc_token(paragraph_text, do_lower_case=do_lower_case),
            orig_answer_text=None,
            start_position=None,
            end_position=None)]

  logging.info("Load {} examples".format(len(examples)))
  features = convert_examples_to_features(examples, tokenizer, max_seq_length,doc_stride, max_query_length, _is_training)
  return examples, features




  
