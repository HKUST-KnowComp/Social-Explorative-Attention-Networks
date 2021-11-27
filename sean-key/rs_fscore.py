import util
import networkx as nx
import ucb
import timeit
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from article_classifier import shan
from keras.preprocessing.sequence import pad_sequences
import time


def metric_evaluate(
        y_true,
        y_pred,
        results):
    Y_pred = (y_pred > 0.4).astype(np.int32)
    y_f1 = f1_score(y_true, Y_pred)
    y_precision = precision_score(y_true, Y_pred)
    y_recall = recall_score(y_true, Y_pred)
    y_acc = accuracy_score(y_true, Y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    y_auc = auc(fpr, tpr)
    print("Y_acc:", y_acc)
    print("Y_auc:", y_auc)
    print("Y_f1:", y_f1)
    print("Y_precision:", y_precision)
    print("Y_recall:", y_recall)
    results["acc"].append(y_acc)
    results["auc"].append(y_auc)
    results["f1"].append(y_f1)
    results["precision"].append(y_precision)
    results["recall"].append(y_recall)
    return results


def result_printing(results):
    # from the 3rd day
    acc_list = results["acc"][1:]
    auc_list = results["auc"][1:]
    f1_list = results["f1"][1:]
    precision_list = results["precision"][1:]
    recall_list = results["recall"][1:]

    avgs = {"acc": 0, "auc": 0, "f1": 0, "precision": 0, "recall": 0}
    for i in range(len(acc_list)):
        print(
            i + 3,
            acc_list[i],
            auc_list[i],
            f1_list[i],
            precision_list[i],
            recall_list[i])
    avgs["acc"] = np.array(acc_list).mean(axis=0)
    avgs["auc"] = np.array(auc_list).mean(axis=0)
    avgs["f1"] = np.array(f1_list).mean(axis=0)
    avgs["precision"] = np.array(precision_list).mean(axis=0)
    avgs["recall"] = np.array(recall_list).mean(axis=0)
    print("average acc:", avgs["acc"])
    print("average auc:", avgs["auc"])
    print("average f1:", avgs["f1"])
    print("average precision:", avgs["precision"])
    print("average recall:", avgs["recall"])
    return avgs


def load_daily_data(sample_dir, day_id):
    sample_dir_keyword = sample_dir + '_' + str(args.max_len_doc) + '_' + str(args.max_len_user)
    out_file_user_keyword = sample_dir_keyword + '/day_user_keyword_' + str(day_id) + '.npy'
    out_file_article_keyword = sample_dir_keyword + '/day_article_keyword_' + str(day_id) + '.npy'
    out_file_label = sample_dir + '/day_label_' + str(day_id) + '.npy'
    out_file_creator = sample_dir + '/day_creator_' + str(day_id) + '.npy'
    out_file_user_index = sample_dir + '/day_user_' + str(day_id) + '.npy'
    out_file_article_index = sample_dir + '/day_article_' + str(day_id) + '.npy'
    users_keyword = np.load(out_file_user_keyword, allow_pickle=True)
    articles_keyword = np.load(out_file_article_keyword, allow_pickle=True)
    labels = np.load(out_file_label, allow_pickle=True)
    creators = np.load(out_file_creator, allow_pickle=True)
    users_index = np.load(out_file_user_index, allow_pickle=True)
    articles_index = np.load(out_file_article_index, allow_pickle=True)
    return users_keyword, articles_keyword, labels, creators, users_index, articles_index


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    edgelist = 'karate.edgelist'

    G = nx.read_edgelist(
        edgelist,
        nodetype=int,
        create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    # G = G.to_undirected()
    print("G is directed:", G.is_directed())
    return G


users_keyword_dict = dict()
articles_keyword_dict = dict()


def build_social_sample(walks, users_keyword, articles_keyword, labels, users_index, articles_index):
    social_samples = []
    for i in range(len(labels)):
        user_id = users_index[i][0]
        article_id = articles_index[i][0]
        for j in range(len(walks[user_id])):
            social_samples.append([
                user_id_to_keyword(walks[user_id][j]),
                articles_keyword[i],
                labels[i],
                i])

        # update user and article keywords
        users_keyword_dict[user_id] = users_keyword[i]
        articles_keyword_dict[article_id] = articles_keyword[i]

    d_vectorized_sample = np.array(social_samples)
    d_user = np.rollaxis(
            np.dstack(list(d_vectorized_sample[:, 0])), -1)
    d_article = pad_sequences(
            d_vectorized_sample[:, 1], maxlen=args.max_len_doc)
    d_label = pad_sequences(d_vectorized_sample[:, 2], maxlen=1)
    d_id = d_vectorized_sample[:, 3]

    return [d_user, d_article, d_label, d_id]


def user_id_to_keyword(users_id):
    users_friends = np.zeros((args.walk_length+1, args.max_len_user))
    for i in range(len(users_id)):
        user = users_id[i]
        if user in users_keyword_dict:
            users_friends[i] = users_keyword_dict[user]
    return users_friends


def personalized_evaluate(
    f1_dict,
    users,
    truth,
    prediction,
):
    prediction = (prediction > 0.4).astype(np.int32)
    truth_dict = {}
    predict_dict = {}
    for i in range(len(users)):
        user = users[i][0]
        if user not in truth_dict:
            truth_dict[user] = []
            predict_dict[user] = []
        truth_dict[user].append(truth[i][0])
        predict_dict[user].append(prediction[i][0])

    for i in range(len(users)):
        user = users[i][0]
        f1_dict[user].append(
            f1_score(
                np.array(
                    truth_dict[user]), np.array(
                    predict_dict[user])))
    return f1_dict


def random_walks(x):
    nodes = list(x.nodes())
    walks = {}
    for node in nodes:
        walks[node] = []
    for walk_iter in range(args.num_walks):
        for node in nodes:
            walk = [node]
            friends = list(np.random.randint(1, args.max_user, size=args.walk_length))
            walk.extend(friends)
            walks[node].append(walk)
    return walks


def sean_training(embedding_matrix):

    def compute_average(ids, y):
        pred = []
        votes = {}
        for i in range(len(ids)):
            j = ids[i]
            if j not in votes:
                votes[j] = []
            votes[j].append(y[i])
        for i in range(len(votes)):
            key = list(votes.keys())[i]
            pred.append([np.mean(votes[key])])
        return np.array(pred)

    # initial the walks for every user
    nx_G = read_graph()
    walks = random_walks(nx_G)
    print("initialing ucb...")
    ucb_G = ucb.Graph(nx_G=nx_G, alpha=args.alpha, init_walks=walks)
    init_num_user_explored, init_acc_history, init_walks = ucb_G.init_all()

    vocab_size = embedding_matrix.shape[0]
    classifier = shan(vocab_size=vocab_size,
                      max_len_user=args.max_len_user,
                      max_len_doc=args.max_len_doc,
                      embedding_dim=args.embedding_dim,
                      hidden_size=args.hidden_size,
                      embedding_matrix=embedding_matrix,
                      use_social=args.use_social,
                      use_sim=args.use_sim,
                      walk_length=args.walk_length+1)
    start_time = timeit.default_timer()
    classifier.build_model()
    stop_time = timeit.default_timer()
    print("compiling time:", stop_time - start_time)
    print("Training SEAN...")
    model = classifier.model

    val_results = {
        "acc": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": []}
    test_results = {
        "acc": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": []}
    prediction = {}
    running_time = {}
    creator_hit = {}

    for day in range(args.day_count - 1):
        print("training day:", day+1, "testing day:", day+2)
        # refresh all dictionaries every 30 days
        if day % 30 == 0:
            ucb_G.num_user_explored = init_num_user_explored
            ucb_G.acc_history = init_acc_history
            ucb_G.walks = random_walks(nx_G)

        # training-validation data
        train_key = day + 1
        train_val_users_keyword, train_val_articles_keyword, train_val_labels, \
        train_val_creators, train_val_users_index, train_val_articles_index = load_daily_data(
            args.processed_feed, train_key)

        indices = np.arange(train_val_users_keyword.shape[0])
        np.random.shuffle(indices)
        train_val_users_keyword = train_val_users_keyword[indices]
        train_val_articles_keyword = train_val_articles_keyword[indices]
        train_val_labels = train_val_labels[indices]
        train_val_users_index = train_val_users_index[indices]
        train_val_articles_index = train_val_articles_index[indices]
        num_validation_samples = int(
            args.validation_split *
            train_val_users_keyword.shape[0])

        train_users_keyword = train_val_users_keyword[:-num_validation_samples]
        train_articles_keyword = train_val_articles_keyword[:-num_validation_samples]
        train_labels = train_val_labels[:-num_validation_samples]
        train_users_index = train_val_users_index[:-num_validation_samples]
        train_articles_index = train_val_articles_index[:-num_validation_samples]
        val_users_keyword = train_val_users_keyword[-num_validation_samples:]
        val_articles_keyword = train_val_articles_keyword[-num_validation_samples:]
        val_labels = train_val_labels[-num_validation_samples:]
        val_users_index = train_val_users_index[-num_validation_samples:]
        val_articles_index = train_val_articles_index[-num_validation_samples:]
        print("train count:", len(train_labels))
        print("validation count:", len(val_labels))

        # training
        print("training...")
        if train_key > 1:
            ucb_G.simulate_nodes = list(set([user[0] for user in train_users_index]))
            # explore friends for training users
            ucb_G.simulate_walks(
                num_walks=args.num_walks,
                walk_length=args.walk_length+1)
            # update the dictionary num_user_explored
            ucb_G.update_freq(ucb_G.walks)
            # users_keyword_dict[user_id] = users_keyword

        train_data = build_social_sample(
            ucb_G.walks, train_users_keyword, train_articles_keyword, train_labels, train_users_index, train_articles_index)
        train_social_inputs = train_data[:2]
        train_outputs = train_data[-2]

        start_time = timeit.default_timer()
        model.fit(train_social_inputs,
                  train_outputs,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  shuffle=True,
                  verbose=1)
        stop_time = timeit.default_timer()
        running_time[train_key] = stop_time - start_time
        print("running time:", running_time[train_key])

        # validation
        print("validating...")
        val_data = build_social_sample(
            ucb_G.walks, val_users_keyword, val_articles_keyword, val_labels, val_users_index, val_articles_index)
        val_social_inputs = val_data[:2]
        y_val = model.predict(
            val_social_inputs,
            batch_size=args.batch_size,
            verbose=1)
        y_val = compute_average(
            val_data[-1], y_val)
        val_results = metric_evaluate(val_labels, y_val, val_results)

        # testing
        print("testing...")
        test_key = day + 2
        test_users_keyword, test_articles_keyword, test_labels, \
        test_creators, test_users_index, test_articles_index = load_daily_data(
            args.processed_feed, test_key)
        test_data = build_social_sample(
            ucb_G.walks, test_users_keyword, test_articles_keyword, test_labels, test_users_index, test_articles_index)
        test_social_inputs = test_data[:2]
        y_test = model.predict(
            test_social_inputs,
            batch_size=args.batch_size,
            verbose=1)
        y_test = compute_average(
            test_data[-1], y_test)
        test_results = metric_evaluate(test_labels, y_test, test_results)

        # update dict accuracy_history after testing
        ucb_G.acc_history = personalized_evaluate(
            ucb_G.acc_history, test_users_index, test_labels, y_test)
        prediction[test_key] = y_test.tolist()
        creator_hit[test_key] = test_creators.tolist()
        print("average y_test shape:", y_test.shape)
        print("user shape:", test_users_index.shape)
    print("validation results:")
    avg_val = result_printing(val_results)
    print("test results:")
    avg_test = result_printing(test_results)

    logs = {
        "param": vars(args),
        "val": val_results,
        "test": test_results,
        "avg_val": avg_val,
        "avg_test": avg_test,
        "running_time": running_time,
        "prediction": prediction,
        "creator_hit": creator_hit}
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = args.output_dir + timestr + '.json'
    with open(output_file_name, "w") as write_file:
        json.dump(logs, write_file)


if __name__ == "__main__":
    start_time = time.time()
    args = util.args
    f_feed_dir = args.processed_feed
    f_emb_matrix = open(args.embedding_matrix, 'rb')
    embed_matrix = np.load(f_emb_matrix)

    print("SEAN running...")
    sean_training(embed_matrix)
    print("--- %s seconds ---" % (time.time() - start_time))

