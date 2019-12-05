import sys
import json
from collections import Counter
import numpy as np


def load_daily_data(sample_dir, day_id):
	out_file_user = sample_dir + 'day_user_' + str(day_id) + '.npy'
	out_file_label = sample_dir + 'day_label_' + str(day_id) + '.npy'
	out_file_creator = sample_dir + 'day_creator_' + str(day_id) + '.npy'
	users = np.load(out_file_user, allow_pickle=True)
	labels = np.load(out_file_label, allow_pickle=True)
	creators = np.load(out_file_creator, allow_pickle=True)
	return users, labels, creators


def gini(array):
	"""Calculate the Gini coefficient of a numpy array."""
	array = array.flatten()
	if np.amin(array) < 0:
		array -= np.amin(array)
	array += 0.0000001
	array = np.sort(array)
	index = np.arange(1, array.shape[0]+1)
	n = array.shape[0]
	return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def gini_calculate(hit):
	hit_all = {}
	for i in range(3, day_count):
		day_id = str(i)
		hit_all[day_id] = []
		for j in range(3, i + 1):
			u_day_id = str(j)
			hit_all[day_id].extend(hit[u_day_id])

	gini_index = []
	for i in range(3, day_count):
		day_id = str(i)
		u_counter = dict(Counter(hit_all[day_id]))
		a = np.array(list(u_counter.values()), dtype=np.float64)
		if sum(a) == 0:
			continue
		gini_index.append(gini(a))
		# print(gini(a))
	return gini_index


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


def main():
	f = open(name)
	d = json.loads(f.read())
	print("test:")
	for key in d["avg_test"]:
		print(key, d["avg_test"][key])

	pred_creator_hit = {}
	true_creator_hit = {}

	pred_user_hit = {}
	true_user_hit = {}
	user_count = []
	for i in range(3, day_count):
		day_id = str(i)
		users, labels, creators = load_daily_data('../data/steemit/processed_feed/', day_id)
		pred_creator_hit[day_id] = []
		true_creator_hit[day_id] = []

		pred_user_hit[day_id] = []
		true_user_hit[day_id] = []
		for j in range(len(d["prediction"][day_id])):
			if d["prediction"][day_id][j][0] > 0.5:
				pred_creator_hit[day_id].append(creators[j])
				pred_user_hit[day_id].append(users[j][0])
			if labels[j][0] > 0.5:
				true_creator_hit[day_id].append(creators[j])
				true_user_hit[day_id].append(users[j][0])
				user_count.append(users[j][0])
		"""
		print "pred"
		dict_u_counter = dict(Counter(pred_user_hit[day_id]))
		print OrderedDict(sorted(dict_u_counter.items()))

		print "true"
		dict_u_counter = dict(Counter(true_user_hit[day_id]))
		print OrderedDict(sorted(dict_u_counter.items()))
		print
		"""

	print("###############################creator#########################")
	pred_creator_gini_index = gini_calculate(pred_creator_hit)
	print("predict_gini:", np.mean(pred_creator_gini_index))
	print("user count:", len(dict(Counter(user_count))))
	"""
	true_creator_gini_index = gini_calculate(true_creator_hit)
	print "true_gini:", np.mean(true_creator_gini_index)

	print "###############################user############################"
	pred_user_gini_index = gini_calculate(pred_user_hit)
	print "predict_gini:", np.mean(pred_user_gini_index)

	true_user_gini_index = gini_calculate(true_user_hit)
	print "true_gini:", np.mean(true_user_gini_index)
	
	"""
	# result_printing(d["test"])
	print(d["param"])


if __name__ == '__main__':
	name = sys.argv[1]
	day_count = int(sys.argv[2])
	main()


