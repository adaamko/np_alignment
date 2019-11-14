import re
import random
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scripts.baseline_utils import process_baseline
from collections import defaultdict


def get_tokenized_text(sen, tokenizer):
    marked_text = "[CLS] " + sen + " [SEP] "
    tokenized_text = tokenizer.tokenize(marked_text)
    return tokenized_text[1:len(tokenized_text)-1]


def tokenize_sentences(sentences, np_to_indices, tokenizer):
    for i,sentence in enumerate(sentences):
        gold = sentence['aligns']
        gold_filtered = []
        for goldalign in gold:
            en = re.findall('\d+', goldalign[0] )
            hu = re.findall('\d+', goldalign[1] )
            gold_filtered.append((str(en[0]), str(hu[0])))
        sentence["aligns_filtered"] = gold_filtered
        sentence_en = []
        sentence_hu = []
        np_to_indices[i]["en_sen"] = []
        np_to_indices[i]["hu_sen"] = []

        en_str_to_tokenize = []
        for token in sentence["en_sen"]:
            if type(token) == tuple:
                sentence_en += get_tokenized_text(" ".join(en_str_to_tokenize), tokenizer)
                en_str_to_tokenize = []
                start_ind = len(sentence_en)
                tokenized_np = get_tokenized_text(" ".join(token[1]), tokenizer)
                end_ind = start_ind + len(tokenized_np) - 1
                np_to_indices[i]["en_sen"].append((token[0], start_ind, end_ind))
                sentence_en += tokenized_np
            else:
                en_str_to_tokenize.append(token)
        sentence_en += get_tokenized_text(" ".join(en_str_to_tokenize), tokenizer)


        hu_str_to_tokenize = []
        for token in sentence["hu_sen"]:
            if type(token) == tuple:
                sentence_hu += get_tokenized_text(" ".join(hu_str_to_tokenize), tokenizer)
                hu_str_to_tokenize = []
                start_ind = len(sentence_hu)
                tokenized_np = get_tokenized_text(" ".join(token[1]), tokenizer)
                end_ind = start_ind + len(tokenized_np) - 1
                np_to_indices[i]["hu_sen"].append((token[0], start_ind, end_ind))

                sentence_hu += tokenized_np
            else:
                hu_str_to_tokenize.append(token)
        sentence_hu += get_tokenized_text(" ".join(hu_str_to_tokenize), tokenizer)

        sentence["sentence_hun"] = sentence_hu
        sentence["sentence_en"] = sentence_en


def get_sentence_embeddings(sentences, np_to_indices, tokenizer, model):
    too_long_sentences = []
    for i,sentence in enumerate(sentences):
        batch_i = 0
        text_hu = sentences[i]["sentence_hun"]
        text_en = sentences[i]["sentence_en"]

        tokenized_text = []
        tokenized_text.append("[CLS]")
        tokenized_text += text_en
        tokenized_text.append("[SEP]")
        tokenized_text += text_hu
        if len(tokenized_text) > 512:
            too_long_sentences.append(i)
            continue

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * (len(sentences[i]["sentence_en"]) + 2) + [1] * len(sentences[i]["sentence_hun"])
        """
        print(tokenized_text)
        print(len(tokenized_text))
        print(len(segments_ids))
        """
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)


        token_embeddings = []

        # For each token in the sentence...
        for token_i in range(len(tokenized_text)):

        # Holds 12 layers of hidden states for each token
            hidden_layers = []

        # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):

            # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]

                hidden_layers.append(vec)

            token_embeddings.append(hidden_layers)

        concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]

        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]

        en_emb = []
        hu_emb = []

        for np in np_to_indices[i]["en_sen"]:
            en_emb.append((np[0], summed_last_4_layers[np[1]+1:np[2]+2]))

        for np in np_to_indices[i]["hu_sen"]:
            hu_emb.append((np[0], summed_last_4_layers[np[1]+len(text_en)+2:np[2]+len(text_en)+3]))
        np_to_indices[i]["en_emb"] = en_emb
        np_to_indices[i]["hu_emb"] = hu_emb


def get_vocabulary(sentences, np_to_indices):
    i = 0
    word2idx = defaultdict(dict)
    voc = []
    for sen in sentences:
        en_sen = []
        hu_sen = []
        indices = np_to_indices[sen['id']]
        for ind in indices['en_sen']:
            words = sen['sentence_en'][ind[1]:ind[2]+1]
            np_i = []
            for w in words:
                np_i.append(str(i))
                voc.append(str(i))
                i+=1

            en_sen.append((ind[0], np_i))

        for ind in indices['hu_sen']:
            words = sen['sentence_hun'][ind[1]:ind[2]+1]
            np_i = []
            for w in words:
                np_i.append(str(i))
                voc.append(str(i))
                i+=1
            hu_sen.append((ind[0], np_i))

        word2idx[sen['id']]["sentence_en"] = en_sen
        word2idx[sen['id']]["sentence_hun"] = hu_sen

    return word2idx, voc


def init_bert_embeddings(np_to_indices):
    bert_weights = []
    for np in np_to_indices:
        for emb in np_to_indices[np]['en_emb']:
            for e in emb[1]:
                bert_weights.append(e.tolist())
        for emb in np_to_indices[np]['hu_emb']:
            for e in emb[1]:
                bert_weights.append(e.tolist())
    return bert_weights


def process():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    sentences = process_baseline("1984.sen-aligned.np-aligned.gold")
    sentences[125]["en_sen"] = [(0, ['Audience'])]
    sentences[125]["hu_sen"] = [(0, ['A', 'hallgatóság'])]
    sentences[125]["aligns"] = [('0', '0')]

    np_to_indices = defaultdict(dict)

    tokenize_sentences(sentences, np_to_indices, tokenizer)
    get_sentence_embeddings(sentences, np_to_indices, tokenizer, model)
    word2idx, voc = get_vocabulary(sentences, np_to_indices)
    voc_to_id = {}
    for i in voc:
        voc_to_id[i] = int(i)
    
    bert_weights = init_bert_embeddings(np_to_indices)

    return sentences, np_to_indices, word2idx, voc, voc_to_id, bert_weights