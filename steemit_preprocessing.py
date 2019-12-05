# -*- coding: utf-8 -*
import timeit
from tqdm import tqdm
import json
import progressbar
from time import sleep
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk import sent_tokenize
import argparse


"""
global variables
"""
# global variables
activity_name = '../dataset/steemit/en/processed_user_activity.json'
relation_name = '../dataset/steemit/en/processed_user_relation.json'
index_name = '../dataset/steemit/en/processed_user_index.json'
article_name = '../dataset/steemit/en/new_article.json'
f_relation = open(relation_name)
user_relation = json.loads(f_relation.read())
f_index = open(index_name)
user_dict = json.loads(f_index.read())
f_article = open(article_name)
article_dict = json.loads(f_article.read())
user_reading = {}
new_article_dict = {}
day_count = 370  # 2017-7-1 - 2018-7-6
user_payout = {}
daily_nx = {}
user_payout_all = {}


def parse_args():
    '''
    Parses the steemit preprocessing arguments.
    '''
    parser = argparse.ArgumentParser(description="Twitter Dataset Preprocessing.")
    parser.add_argument('--word-representation', nargs='?', default='../dataset/steemit/word2vec.txt',
                        help='Word representation path')
    parser.add_argument('--embedding-dim', type=int, default=300,
                        help='Number of word embedding dimensions. Default is 300.')
    parser.add_argument('--max_num_sent', type=int, default=30,
                        help='Max number of sentences in feed. Default is 30.')  # 87%
    parser.add_argument('--max_len_sent', type=int, default=100,  # 99%
                        help='Max length for a sentence. Default is 100.')
    parser.add_argument('--feed-output', nargs='?', default='data/steemit/processed_feed/',
                        help='Output for processed_feed.')
    parser.add_argument('--emb-output', nargs='?', default='embedding_matrix.npy',
                        help='Pre-trained word embedding matrix.')
    return parser.parse_args()


def word_tokenize(sent):
    tokens = nltk.word_tokenize(sent)
    new_tokens = []
    for token in tokens:
        if token != '.' and token[-1] in ['.']:
            token = token[:-1]
        new_tokens.append(token)
    return new_tokens


def tokenize_paragraph(paragraph):
    raw_sentences = sent_tokenize(paragraph)
    sentences = []
    for raw_sentence in raw_sentences:
        tokens = word_tokenize(raw_sentence)
        sentences.append(tokens)
    return sentences


def padding_article(text, args):
    pad_text = pad_sequences(
        text, maxlen=args.max_len_sent)
    pad_num_sent = args.max_num_sent - len(text)
    if pad_num_sent > 0:
        completion = np.zeros(
            shape=(
                pad_num_sent,
                args.max_len_sent),
            dtype=int)
        pad_text = np.concatenate((completion, pad_text))
    else:
        pad_text = pad_text[:args.max_num_sent, :]
    return pad_text


def word_to_vec(args):
    embeddings_index = {}
    f = open(args.word_representation)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def print_sample(d):
    y = d[100]
    print("user_index:", np.array(y[0]).shape)
    print("article:", np.array(y[1]).shape)
    print("label:", np.array(y[2]).shape)


def build_social_graph():
    user_list = list(user_reading.keys())
    print("user_list length:", len(user_list))
    relation_list = []
    for i in tqdm(range(len(user_reading))):
        user_name = list(user_reading.keys())[i]
        for friend_name in user_relation[user_name]:
            relation_list.append([user_dict[user_name], user_dict[friend_name]])
    np.savetxt('karate.edgelist', np.array(relation_list), fmt='%d')


def build_sample():
    """
    create positive and negative samples
    :return: user_reading, article_dict
    """
    with open(activity_name, "r") as f:
        print("loading activity file...")
        activity_list = json.loads(f.read())
        # positive sample
        for i in tqdm(range(len(activity_list))):
            d = activity_list[i]
            """
            [u'dflo', 379, 56603032,
             "Congratulate me please Congratulations to me.",
             'gbemy', '0.0', 1]
            """
            if len(d[3]) == 0:
                continue
            user_name = d[0]
            day_id = int(d[1])
            article_id = d[2]
            creator = d[4]
            payout = float(d[5])
            if user_name not in user_reading.keys():
                user_reading[user_name] = {}
            if day_id not in user_reading[user_name].keys():
                user_reading[user_name][day_id] = {"positive": [], "negative": []}
            user_reading[user_name][day_id]["positive"].append(article_id)

            if user_name not in user_payout.keys():
                user_payout[user_name] = {}
            if day_id not in user_payout[user_name].keys():
                user_payout[user_name][day_id] = []
            user_payout[user_name][day_id].append(payout)

            if day_id not in daily_nx:
                daily_nx[day_id] = []
            daily_nx[day_id].append([user_dict[user_name], user_dict[creator]])
        daily_nx_all = {}
        for day_id in daily_nx:
            daily_nx_all[day_id] = []
            for u_day_id in range(1, day_id):
                daily_nx_all[day_id].extend(daily_nx[u_day_id])
            out_file_nx = opt.feed_output + 'day_nx_' + str(day_id)
            np.savetxt(out_file_nx, daily_nx_all[day_id], fmt='%d')
        build_social_graph()
        # negative sample
        for user_name in user_reading:
            day_sorted = sorted(user_reading[user_name].keys())
            for day_id in day_sorted:
                for friend_name in user_relation[user_name]:
                    if friend_name in user_reading:
                        if day_id in user_reading[friend_name]:
                            for article_id in user_reading[friend_name][day_id]["positive"]:
                                if article_id not in user_reading[user_name][day_id]["positive"]:
                                    user_reading[user_name][day_id]["negative"].append(article_id)


def create_vocab():
    def update_vocab(x):
        x = x.split(' ')
        for word in x:
            if word in vocab:
                continue
            vocab[word] = len(vocab)
    vocab = {
        "": 0,
    }
    # add progress bar
    maxval = len(article_dict) + 1
    bar = progressbar.ProgressBar(
        maxval=maxval, widgets=[
            progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])
    progress = 0
    bar.start()
    for key in article_dict:
        # progress bar
        bar.update(progress + 1)
        progress += 1
        sleep(0.00001)
        update_vocab(str(article_dict[key]['text']))
    bar.finish()
    print('vocab count:', len(vocab))
    return vocab


def sample_format(args, vocab):
    social_negative_count = 0
    social_positive_count = 0

    # article to vector
    def map_vocab(x, x_vocab):
        if x in x_vocab:
            return x_vocab[x]
        else:
            return x_vocab['']
    sent_len_list = []
    sent_num_list = []
    # add progress bar
    maxval = len(article_dict) + 1
    bar = progressbar.ProgressBar(
        maxval=maxval, widgets=[
            progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])
    progress = 0
    bar.start()

    for key in article_dict:
        # progress bar
        bar.update(progress + 1)
        progress += 1
        sleep(0.000001)
        text = str(article_dict[key]['text']).strip()
        sents = tokenize_paragraph(text)
        sent_num_list.append(len(sents))
        sent_vectors = []
        for sent in sents:
            sent_len_list.append(len(sent))
            sent_vector = []
            for word in sent:
                sent_vector.append(map_vocab(word, vocab))
            sent_vectors.append(sent_vector)
        if len(sent_vectors) > 0:
            new_article_dict[key] = sent_vectors
    bar.finish()

    for key in new_article_dict:
        new_article_dict[key] = padding_article(new_article_dict[key], args)

    social_sample_data = {}
    vectorized_data = {}
    for i in tqdm(range(day_count)):
        day_id = i + 1
        social_sample_data[day_id] = []
        user_payout_all[day_id] = {}
        for user_name in user_reading:
            if day_id not in user_reading[user_name].keys():
                continue
            user_id = user_dict[user_name]
            user_payout_all[day_id][user_id] = 0
            for u_day_id in range(1, day_id):
                if u_day_id in user_payout[user_name]:
                    user_payout_all[day_id][user_id] += sum(user_payout[user_name][u_day_id])
            user_index = user_dict[user_name]
            # positive sample format
            if len(user_reading[user_name][day_id]["positive"]) > 0:
                for article_id in user_reading[user_name][day_id]["positive"]:
                    if article_id in new_article_dict:
                        social_sample = [
                            [user_index],
                            new_article_dict[article_id],
                            user_dict[article_dict[article_id]['author']],
                            [1]]
                        social_sample_data[day_id].append(social_sample)
                        social_positive_count += 1
        
            # negative sample format
            if len(user_reading[user_name][day_id]["negative"]) > 0:
                for article_id in user_reading[user_name][day_id]["negative"]:
                    if article_id in new_article_dict:
                        social_sample = [
                            [user_index],
                            new_article_dict[article_id],
                            user_dict[article_dict[article_id]['author']],
                            [0]]
                        social_sample_data[day_id].append(social_sample)
                        social_negative_count += 1
        # padding
        d_vectorized_sample = np.array(social_sample_data[day_id])
        d_user_index = pad_sequences(d_vectorized_sample[:, 0], maxlen=1)
        d_article = np.rollaxis(np.dstack(list(d_vectorized_sample[:, 1])), -1)
        d_creator = d_vectorized_sample[:, 2]
        d_label = pad_sequences(d_vectorized_sample[:, 3], maxlen=1)
        vectorized_data[day_id] = [d_user_index, d_article, d_label]
        out_file_user = args.feed_output + 'day_user_' + str(day_id)
        out_file_article = args.feed_output + 'day_article_' + str(day_id)
        out_file_creator = args.feed_output + 'day_creator_' + str(day_id)
        out_file_label = args.feed_output + 'day_label_' + str(day_id)
        np.save(out_file_user, d_user_index)
        np.save(out_file_article, d_article)
        np.save(out_file_creator, d_creator)
        np.save(out_file_label, d_label)
    with open('user_payout.json', 'w') as outfile:
        json.dump(user_payout_all, outfile)
    return vectorized_data


def get_embed(vocab, args):
    embeddings_index = word_to_vec(args)
    vocab_size = len(vocab) + 1
    emb_matrix = np.zeros((vocab_size, args.embedding_dim))

    # create progress bar
    maxval = vocab_size
    bar = progressbar.ProgressBar(
        maxval=maxval, widgets=[
            progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])
    progress = 0
    bar.start()

    # unk_embedding = np.zeros(param_zh.embedding_dim)  # <'unk'>
    for word, i in vocab.items():

        # progress bar
        bar.update(progress)
        progress += 1
        sleep(0.00001)

        if word in embeddings_index:
            embedding_vector = embeddings_index[word]
        else:
            embedding_vector = None
        if embedding_vector is not None:
            emb_matrix[i] = embedding_vector
        else:
            emb_matrix[i] = np.zeros(args.embedding_dim)
    emb_matrix[0] = np.zeros(args.embedding_dim)
    bar.finish()
    return emb_matrix


if __name__ == '__main__':
    start = timeit.default_timer()
    opt = parse_args()
    print(opt)
    print("Building sample...")
    build_sample()
    print("Building vocabulary...")
    vocabs = create_vocab()
    print("Building embedding matrix...")
    embedding_matrix = get_embed(vocabs, opt)
    print("Saving embedding_matrix....")
    np.save(opt.emb_output, embedding_matrix)
    print("Building and saving vectorized data...")
    processed_data = sample_format(opt, vocabs)
    print_sample(processed_data)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
