from flask import Flask, request
from langdetect import detect, DetectorFactory
import numpy as np
import sys
import torch
import bclm
from utils.data import Data
from model.seqlabel import SeqLabel

params = {'status': 'decode'}
params['load_model_dir'] = 'data/token.char_cnn.ft_tok.46_seed.104.model'
params['dset_dir'] = 'data/token.char_cnn.ft_tok.46_seed.dset'

data = Data()
data.status = "decode"
data.load(params['dset_dir'])
data.load_model_dir = "data/token.char_cnn.ft_tok.46_seed.104.model"
data.dset_dir = "data/token.char_cnn.ft_tok.46_seed.dset"
model = SeqLabel(data)
model.load_state_dict(torch.load(data.load_model_dir, map_location=torch.device('cpu')))

def tokenize_text(text):
    sents = []
    for line in text.split('\n'):
        if line.strip():
            toks = bclm.tokenize(line.rstrip())
            sents.append(toks)
    return sents


def write_tokens_file(sents, dummy_o=False, only_tokens=False):
    tagged_words = []
    for sent in sents:
        for fields in sent:
            if type(fields) is str:
                word = fields
            else:
                word = fields[0]
            if only_tokens:
                line = word
            elif dummy_o:
                line = word + ' O'
                tagged_words.append(line)
            else:
                line = word + ' ' + fields[-1]
                tagged_words.append(line)

    return tagged_words


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet= data.word_alphabet, char_alphabet=data.char_alphabet, feature_alphabets=data.feature_alphabets
                  ,label_alphabet=data.label_alphabet, number_normalized=data.number_normalized,
                  max_sent_length=250, sentence_classification=False, split_token='\t', char_padding_size=-1,
                  char_padding_symbol='</pad>'):
    feature_alphabets = []
    feature_num = len(feature_alphabets)
    in_lines = input_file
    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []

    ## if sentence classification data format, splited by \t
    if sentence_classification:
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split(split_token)
                sent = pairs[0]
                if sys.version_info[0] < 3:
                    sent = sent.decode('utf-8')
                original_words = sent.split()
                for word in original_words:
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    word_Ids.append(word_alphabet.get_index(word))
                    ## get char
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                        assert (len(char_list) == char_padding_size)
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)

                label = pairs[-1]
                label_Id = label_alphabet.get_index(label)
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx + 1].split(']', 1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                ## combine together and return, notice the feature/label as different format with sequence labeling task
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                    instence_texts.append([words, feat_list, chars, label])
                    instence_Ids.append([word_Ids, feat_Id, char_Ids, label_Id])
                words = []
                features = []
                chars = []
                char_Ids = []
                word_Ids = []
                feature_Ids = []
                label_Ids = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
            instence_texts.append([words, feat_list, chars, label])
            instence_Ids.append([word_Ids, feat_Id, char_Ids, label_Id])
            words = []
            features = []
            chars = []
            char_Ids = []
            word_Ids = []
            feature_Ids = []
            label_Ids = []

    else:
        ### for sequence labeling data format i.e. CoNLL 2003
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                words.append(word)
                if number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                labels.append(label)
                word_Ids.append(word_alphabet.get_index(word))
                label_Ids.append(label_alphabet.get_index(label))
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx + 1].split(']', 1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                features.append(feat_list)
                feature_Ids.append(feat_Id)
                ## get char
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                    assert (len(char_list) == char_padding_size)
                else:
                    ### not padding
                    pass
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
            else:
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                    instence_texts.append([words, features, chars, labels])
                    instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids])
                words = []
                features = []
                chars = []
                labels = []
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                label_Ids = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
            instence_texts.append([words, features, chars, labels])
            instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids])
            words = []
            features = []
            chars = []
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def batchify_with_label(input_batch_list, gpu, if_train=True, sentence_classification=False):
    return batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train)


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
                         mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover,
                  sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert (len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def evaluate(data, model, nbest=None):


    instances = data.raw_Ids
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
            instance, data.HP_gpu, False, data.sentence_classification)
        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:, :, 0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                            mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover,
                                               data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label


    if nbest and not data.sentence_classification:
        return  nbest_pred_results, pred_scores
    return  pred_results, pred_scores


def load_model_decode(data):
    pred_results, pred_scores = evaluate(data, model,  data.nbest)
    return pred_results, pred_scores


def prepare_output(words):
    multi_word = {"word":"",
                  "tag":[]}
    final_dict = {"Person": [],
                  "Organization": [],
                  "Geo-Political": [],
                  "Location": [],
                  "Facility": [],
                  "Event": [],
                  "Product": [],
                  "Language": []}
    for w in words:

        word = w.split(" ")[0]
        tag = w.split(" ")[1]
        if tag == "O" and len(multi_word["tag"])!= 0:
            if multi_word["tag"][0] == "PER":
                final_dict["Person"].append(multi_word["word"])
            elif multi_word["tag"][0] == "ORG":
                final_dict["Organization"].append(multi_word["word"])
            elif multi_word["tag"][0] == "GPE":
                final_dict["Geo-Political"].append(multi_word["word"])
            elif multi_word["tag"][0] == "LOC":
                final_dict["Location"].append(multi_word["word"])
            elif multi_word["tag"][0] == "FAC":
                final_dict["Facility"].append(multi_word["word"])
            elif multi_word["tag"][0] == "EVE":
                final_dict["Event"].append(multi_word["word"])
            elif multi_word["tag"][0] == "DUC":
                final_dict["Product"].append(multi_word["word"])
            elif multi_word["tag"][0] == "ANG":
                final_dict["Language"].append(multi_word["word"])
            multi_word = {"word": "",
                          "tag": []}
        if tag != "O":
            tag = tag.split("-")
            if tag[0] == "S":
                if tag[1] == "PER":
                    final_dict["Person"].append(word)
                elif tag[1] == "ORG":
                    final_dict["Organization"].append(word)
                elif tag[1] == "GPE":
                    final_dict["Geo-Political"].append(word)
                elif tag[1] == "LOC":
                    final_dict["Location"].append(word)
                elif tag[1] == "FAC":
                    final_dict["Facility"].append(word)
                elif tag[1] == "EVE":
                    final_dict["Event"].append(word)
                elif tag[1] == "DUC":
                    final_dict["Product"].append(word)
                elif tag[1] == "ANG":
                    final_dict["Language"].append(word)
            if tag[0] == "B":
                multi_word["word"] = multi_word["word"] +word + " "
                multi_word["tag"].append(tag[1])
            if tag[0] == "I":
                multi_word["word"] = multi_word["word"] + word  + " "
            if tag[0] == "E":
                multi_word["word"] =  multi_word["word"] + word
                if multi_word["tag"][0] == "PER":
                    final_dict["Person"].append(multi_word["word"])
                elif multi_word["tag"][0] == "ORG":
                    final_dict["Organization"].append(multi_word["word"])
                elif multi_word["tag"][0] == "GPE":
                    final_dict["Geo-Political"].append(multi_word["word"])
                elif multi_word["tag"][0] == "LOC":
                    final_dict["Location"].append(multi_word["word"])
                elif multi_word["tag"][0] == "FAC":
                    final_dict["Facility"].append(multi_word["word"])
                elif multi_word["tag"][0] == "EVE":
                    final_dict["Event"].append(multi_word["word"])
                elif multi_word["tag"][0] == "DUC":
                    final_dict["Product"].append(multi_word["word"])
                elif multi_word["tag"][0] == "ANG":
                    final_dict["Language"].append(multi_word["word"])
                multi_word = {"word": "",
                              "tag": []}
    return final_dict



app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/predict', methods=['POST'])
def prepare_text():
    # get sent from user online
    request_data = request.get_json()

    input_sent = request_data['text']


    # Detect language of text
    try:
        DetectorFactory.seed = 0
        lang = detect(input_sent)
    except:
        return "No text enter", 400

    if lang != "he":
        return "the language is not Hebrew, please type Hebrew language to detect topic.", 400

    if not input_sent:
        return 400

    if len(input_sent) < 7:
        return "your text is too small, please type more words to detect", 400

    # preprocessing sent
    # remove all char except Heb words
    sents = tokenize_text(input_sent)
    tagged_words = write_tokens_file(sents, dummy_o=True)
    data.raw_dir = tagged_words
    data.raw_texts, data.raw_Ids = read_instance(tagged_words)

    decode_results, pred_scores = load_model_decode(data)
    final_tagged = data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')

    final = prepare_output(final_tagged)

    deleted_keys = []
    for ner in final.keys():
        if len(final[ner]) == 0:
            deleted_keys.append(ner)

    for j in deleted_keys:
        del final[j]
    # Return on a JSON format
    return final


@app.route('/check', methods=['GET'])
def check():
    return "every things right! "


if __name__ == '__main__':
    app.run( host='0.0.0.0')
