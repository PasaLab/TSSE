# -*- coding: utf-8 -*-

from queue import Queue


class voseAlias:
    def __init__(self, n):
        self.n = n  # dimension
        self.pw = [0.0] * n  # proportions
        self.sum_pw = 0.0  # sum of proportions

        self.table = [[0, 0]] * n  # alias probabilities and indices
        self.num_samples = -1  # number of samples between replenishment

    def construct_table(self):
        self.table = [[0, 0]] * self.n

        # 1. Multiply each probability by n.
        p = [i * self.n / self.sum_pw for i in self.pw]

        # 2. Create two worklists, Small and Large.
        small_que, large_que = Queue(), Queue()

        # 3. For each scaled probability pi:
        #      a. If pi<1, add i to Small.
        #      b. Otherwise(pi≥1), add i to Large.
        for i in range(self.n):
            if p[i] < 1:
                small_que.put(i)
            else:
                large_que.put(i)

        # 4. While Small and Large are not empty : (Large might be emptied first)
        #       a. Remove the first element from Small; call it l.
        #       b. Remove the first element from Large; call it g.
        #       c. Set Prob[l] = pl.
        #       d. Set Alias[l] = g.
        #       e. Set pg : = (pl + pg)−1. (This is a more numerically stable option.)
        #       f. If pg<1, add g to Small.
        #       g. Otherwise(pg≥1), add g to Large.
        while not (small_que.empty() or large_que.empty()):
            l = small_que.get()
            g = large_que.get()
            self.table[l][0] = p[l]  # Prob[l] = p[l];
            self.table[l][1] = g  # Alias[l] = g;
            p[g] = (p[g] + p[l]) - 1
            if p[g] < 1:
                small_que.put(g)
            else:
                large_que.put(g)

        # 5. While Large is not empty :
        #       a. Remove the first element from Large; call it g.
        #       b. Set Prob[g] = 1.
        while not large_que.empty():
            g = large_que.get()
            self.table[g][0] = 1  # Prob[g] = 1;

        # 6. While Small is not empty : This is only possible due to numerical instability.
        #       a. Remove the first element from Small; call it l.
        #       b. Set Prob[l] = 1.
        while not small_que.empty():
            l = small_que.get()
            self.table[l][0] = 1  # Prob[g] = 1

        self.num_samples = 1
        del p

    def sample(self, fair_die, u):
        res = u < self.table[fair_die][0]
        return fair_die if res else self.table[fair_die][1]
