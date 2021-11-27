# -*- coding: utf-8 -*
import timeit
from keywords_extraction import extract_tags
import codecs
from tqdm import tqdm
import argparse
import json
from time import sleep

import numpy as np
import progressbar
from keras.preprocessing.sequence import pad_sequences

import multiprocessing as mp
from itertools import repeat

"""
global variables
"""
# global variables
activity_name = '../dataset/steemit/processed_user_activity.json'
relation_name = '../dataset/steemit/processed_user_relation.json'
index_name = '../dataset/steemit/processed_user_index.json'
article_name = '../dataset/steemit/new_article.json'
f_relation = open(relation_name)
user_relation = json.loads(f_relation.read())
f_index = open(index_name)
user_dict = json.loads(f_index.read())
f_article = open(article_name)
article_dict = json.loads(f_article.read())
user_reading = {}
new_article_dict = {}
article_keywords_dict ={}
day_count = 370  # 2017-7-1 - 2018-7-6


def parse_args():
    '''
    Parses the steemit preprocessing arguments.
    '''
    parser = argparse.ArgumentParser(
        description="Steemit Dataset Preprocessing.")
    parser.add_argument(
        '--word-representation',
        nargs='?',
        default='../word2vec.txt',
        help='Word representation path')
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=300,
        help='Number of word embedding dimensions. Default is 300.')
    parser.add_argument(
        '--max-len-doc',
        type=int,
        default=90,
        help='Max number of keywords to represent an article. Default is 90.')
    parser.add_argument(
        '--max-len-user',
        type=int,
        default=200,
        help='Max number of keywords to represent a user. Default is 200.')
    parser.add_argument('--feed-output', nargs='?', default='processed_feed/',
                        help='Output for processed_feed.')
    parser.add_argument(
        '--emb-output',
        nargs='?',
        default='embedding_matrix.npy',
        help='Pre-trained word embedding matrix.')
    return parser.parse_args()


def article_id_list_to_text(id_list):
    if len(id_list) > 10:
        id_list = id_list[-10:]
    text = ""
    for y in id_list:
        str_ = str(article_dict[y]['text'])
        text = text + " " + str_
    sentence_keywords = extract_tags(text, num_key=opt.max_len_user)[0]
    x = ",".join(sentence_keywords)
    return x


def article_id_to_text(target_id):
    text = str(article_dict[target_id]['text'])
    sentence_keywords = extract_tags(
        text, num_key=opt.max_len_doc)[0]
    x = ",".join(sentence_keywords)
    return x


def create_vocab(data, init_vocab=None):
    if not init_vocab:
        vocab = {
            "": 0,
        }

    else:
        vocab = init_vocab

    def update_vocab(x):
        x = x.split(',')
        for word in x:
            if word in vocab:
                continue
            vocab[word] = len(vocab)
    # add progress bar
    maxval = day_count
    bar = progressbar.ProgressBar(
        maxval=maxval, widgets=[
            progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])
    progress = 0
    bar.start()
    for day in data:
        # progress bar
        bar.update(progress + 1)
        progress += 1
        sleep(0.00001)
        for i in range(len(data[day])):
            update_vocab(data[day][i][0])
            update_vocab(data[day][i][1])
    bar.finish()
    vocab[" "] = 0
    print(len(vocab))  # 85417
    return vocab


def print_sample(d):
    y = d[100]
    print("user:", np.array(y[0]).shape)
    print("article:", np.array(y[1]).shape)
    print("label:", np.array(y[2]).shape)
    print("creator:", np.array(y[3].shape))
    print("user_index:", np.array(y[4].shape))
    print("article_index:", np.array(y[5].shape))


def text_to_digit(vocab, text):
    # 0
    def map_vocab(x):
        if x in vocab:
            return vocab[x]
        else:
            return vocab['']
    res = []
    text = text.split(',')
    for word in text:
        digit = map_vocab(word)
        if digit > 0:
            res.append(digit)
    return res


def word_to_vec():
    embeddings_index = {}
    f = open(opt.word_representation)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def get_embed():
    embeddings_index = word_to_vec()
    vocab_size = len(vocabs) + 1
    embedding_matrix = np.zeros((vocab_size, opt.embedding_dim))

    # create progress bar
    maxval = vocab_size
    bar = progressbar.ProgressBar(
        maxval=maxval, widgets=[
            progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])
    progress = 0
    bar.start()

    unk_embedding = np.zeros(opt.embedding_dim)  # <'unk'>
    for word, i in vocabs.items():

        # progress bar
        bar.update(progress)
        progress += 1
        sleep(0.00001)

        if word in embeddings_index:
            embedding_vector = embeddings_index[word]
        else:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = unk_embedding

    embedding_matrix[0] = np.zeros(opt.embedding_dim)
    bar.finish()
    return embedding_matrix


def build_sample():
    global_sample_data = {}
    negative_count = 0
    positive_count = 0

    with codecs.open(activity_name, "r", encoding="utf-8-sig") as f:
        print("loading activity file...")
        activity_list = json.loads(f.read())

        # positive sample
        for i in tqdm(range(len(activity_list))):
        # for i in tqdm(range(2000)):
            d = activity_list[i]
            if len(d[3]) == 0:
                continue
            user_name = d[0]
            day_id = int(d[1])
            article_id = d[2]
            article_keywords_dict[article_id] = article_id_to_text(article_id)
            if user_name not in user_reading.keys():
                user_reading[user_name] = {}
            if day_id not in user_reading[user_name].keys():
                user_reading[user_name][day_id] = {
                    "positive": [], "negative": []}
            user_reading[user_name][day_id]["positive"].append(
                article_id)

    user_list = list(user_reading.keys())
    # add reading history: positive samples
    for i in tqdm(range(len(user_list))):
        user_name = user_list[i]
        day_sorted = sorted(user_reading[user_name].keys())
        for day_id in day_sorted:
            user_reading[user_name][day_id]["reading_history"] = []
            user_reading[user_name][day_id]["reading_keywords"] = ""
            for date in range(1, day_id):
                if date in user_reading[user_name].keys():
                    tmp = user_reading[user_name][date]["positive"]
                    user_reading[user_name][day_id]["reading_history"].extend(
                        tmp)
            if len(user_reading[user_name][day_id]["reading_history"]) > 0:
                user_reading[user_name][day_id]["reading_keywords"] = article_id_list_to_text(
                    user_reading[user_name][day_id]["reading_history"])

    # add reading history: negative samples
    for user_name in user_reading:
        day_sorted = sorted(user_reading[user_name].keys())
        for day_id in day_sorted:
            for friend_name in user_relation[user_name]:
                if friend_name in user_reading:
                    if day_id in user_reading[friend_name]:
                        for article_id in user_reading[friend_name][day_id]["positive"]:
                            if article_id not in user_reading[user_name][day_id]["positive"]:
                                user_reading[user_name][day_id]["negative"].append(
                                    article_id)

    for i in tqdm(range(day_count)):
        day_id = i + 1
        global_sample_data[day_id] = []
        for user_name in user_reading:
            if day_id not in user_reading[user_name].keys():
                continue
            reading_keywords = user_reading[user_name][day_id]["reading_keywords"]

            if len(user_reading[user_name][day_id]["positive"]) > 0:
                for article_id in user_reading[user_name][day_id]["positive"]:
                    global_sample = [
                        reading_keywords,
                        article_keywords_dict[article_id],
                        user_dict[article_dict[article_id]['author']],
                        1,
                        user_dict[user_name],
                        article_id
                    ]
                    global_sample_data[day_id].append(global_sample)
                    positive_count += 1

            if len(user_reading[user_name][day_id]["negative"]) > 0:
                for article_id in user_reading[user_name][day_id]["negative"]:
                    global_sample = [
                        reading_keywords,
                        article_keywords_dict[article_id],
                        user_dict[article_dict[article_id]['author']],
                        0,
                        user_dict[user_name],
                        article_id
                    ]
                    global_sample_data[day_id].append(global_sample)
                    negative_count += 1
    """
    correct_global_sample_count = 0
    for _ in global_sample_data[16]:
        correct_global_sample_count += 1
    """
    return global_sample_data


def sample_format(data):
    vectorized_data = {}

    for j in tqdm(range(day_count)):
        day_id = j + 1
        vectorized_sample = []

        for i in range(len(data[day_id])):
            if data[day_id][i][3] == 0:
                label = [0]
            else:
                label = [1]
            user_keyword_vector = text_to_digit(
                vocabs, data[day_id][i][0])  # max length = 200
            article_keyword_vector = text_to_digit(
                vocabs, data[day_id][i][1])  # max_length = 90
            creator = data[day_id][i][2]
            vectorized_sample.append(
                [user_keyword_vector, article_keyword_vector, creator, label, data[day_id][i][4], data[day_id][i][5]])

        d_vectorized_sample = np.array(vectorized_sample)
        d_user_keyword = pad_sequences(d_vectorized_sample[:, 0], maxlen=opt.max_len_user)
        d_article_keyword = pad_sequences(d_vectorized_sample[:, 1], maxlen=opt.max_len_doc)
        d_creator = d_vectorized_sample[:, 2]
        d_label = pad_sequences(d_vectorized_sample[:, 3], maxlen=1)
        d_user_index = d_vectorized_sample[:, 4]
        d_article_index = d_vectorized_sample[:, 5]
        vectorized_data[day_id] = [d_user_keyword, d_article_keyword, d_creator, d_label, d_user_index, d_article_index]

        out_file_user_keyword = opt.feed_output + 'day_user_keyword_' + str(day_id)
        out_file_article_keyword = opt.feed_output + 'day_article_keyword_' + str(day_id)
        out_file_label = opt.feed_output + 'day_label_' + str(day_id)
        out_file_creator = opt.feed_output + 'day_creator_' + str(day_id)
        out_file_user_index = opt.feed_output + 'day_user_index_' + str(day_id)
        out_file_article_index = opt.feed_output + 'day_article_index_' + str(day_id)
        np.save(out_file_user_keyword, d_user_keyword)
        np.save(out_file_article_keyword, d_article_keyword)
        np.save(out_file_label, d_label)
        np.save(out_file_creator, d_creator)
        np.save(out_file_user_index, d_user_index)
        np.save(out_file_article_index, d_article_index)
    return vectorized_data


if __name__ == '__main__':
    start = timeit.default_timer()
    opt = parse_args()
    print(opt)
    print("Building sample...")
    sample_data = build_sample()
    print("Building vocabulary...")
    vocabs = create_vocab(sample_data, init_vocab=None)
    print("Building and saving vectorized data...")
    processed_data = sample_format(sample_data)
    print_sample(processed_data)
    print("Building embedding matrix...")
    embed_matrix = get_embed()
    print("Saving embedding_matrix....")
    np.save(opt.emb_output, embed_matrix)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
