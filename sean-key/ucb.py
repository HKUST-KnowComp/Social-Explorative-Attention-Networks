import numpy as np
import copy


class Graph():
    def __init__(self, nx_G, alpha, init_walks):
        self.G = nx_G
        self.alpha = alpha
        self.walks = init_walks
        self.acc_history = {}
        self.simulate_nodes = None
        self.num_user_explored = {}  # time t-1

    def init_all(self):
        for node in self.G.nodes():
            self.num_user_explored[node] = 0   # avoid zero
            self.acc_history[node] = []
        for key in self.walks:
            for walk in self.walks[key]:
                for item in walk:
                    self.num_user_explored[item] += 1
        return self.num_user_explored, self.acc_history, self.walks

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate bandit walks from each node.
        '''

        for node in self.simulate_nodes:
            self.walks[node] = self.bandit_walk(
                walk_length=walk_length,
                start_node=node,
                num_walks=num_walks)

    def bandit_walk(self, walk_length, start_node, num_walks):
        walk_score_list = np.array([[[start_node], 0]])
        cur_length = 1
        while cur_length < walk_length:
            cur_length += 1
            candidates = []
            for i in range(len(walk_score_list)):
                walk_score = walk_score_list[i]
                candidates.extend(self.decide_candidate(walk_score))
            candidates = delete_duplicate(candidates)
            d_candidates = np.array(candidates)
            if len(d_candidates) == 0:
                break
            if len(d_candidates) < num_walks:
                walk_score_list = d_candidates
            else:
                ind = np.argsort(d_candidates[:, -1])[-num_walks:]
                walk_score_list = d_candidates[ind]
        walk_list = walk_score_list[:, 0].tolist()
        return walk_list

    def decide_candidate(self, walk_score):
        """
        the decision is based on first-order
        :param walk_score: current walk and score tuple
        :return: list of candidates with current walk and score
        """
        previous_walk = walk_score[0]
        score_sum = walk_score[1]
        walk_score_candidates = []

        cur = previous_walk[-1]
        nbrs = self.G.neighbors(cur)

        for nbr in nbrs:
            acc_nbr = np.mean(self.acc_history[nbr]) if len(self.acc_history[nbr]) else 0
            s = acc_nbr + self.alpha * np.sqrt(
                (np.log(self.num_user_explored[cur])) / float(self.num_user_explored[nbr] + 1))  # avoid zero
            current_walk = copy.copy(previous_walk)
            current_walk.append(nbr)
            tmp = [current_walk, s+score_sum]
            walk_score_candidates.append(tmp)

        return walk_score_candidates

    def update_freq(self, walks):
        for node in self.simulate_nodes:
            for i in range(len(walks[node])):
                friends = walks[node][i][1:]  # walk = [u_i, f_i_1, f_i_2, ....]
                for friend in friends:
                    self.num_user_explored[friend] += 1


def delete_duplicate(candidates):
    tuple_candidates = []
    for x in candidates:
        x_0 = copy.copy(x[0])
        x_0.append(x[1])
        tuple_candidates.append(tuple(x_0))
    unique_tuple_candidates = list(set(tuple_candidates))

    unique_candidates = []
    for item in unique_tuple_candidates:
        y_0 = list(item[:-1])
        y_1 = item[-1]
        unique_candidates.append([y_0, y_1])
    return unique_candidates

