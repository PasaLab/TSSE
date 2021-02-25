# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import multiprocessing
import collections
from collections import defaultdict
from src.utils.common import judgeWordC, judgeWordE, judgeWordEBert
from src.utils.timetest import exe_time
from gensim.models import Word2Vec
from progressbar import *
from src.model.voseAlias import voseAlias
from random import randint

np.seterr(invalid='ignore')
widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]


class TSSE_DMM:
    def __init__(self, alpha, beta, K, num_iters, threshold_wq, cont,
                 vector_path, threshold_sim, filter_size, weight, mh, language):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.num_iters = num_iters
        self.threshold_wq = threshold_wq
        self.cont = cont

        self.times = []  
        self.time_ndocs = []  
        self.corpus = []  
        self.docs_word_ids = []  
        self.docs_pos_ids = []  
       
        self.docs_occ_idxes = []

        self.vocas = []  # id to word
        self.vocas_ids = dict()  # word to id
        self.word_freq = []  # word id counts
        self.M = 0  # number of documents
        self.V = 0  # number of vocabulary
        self.N = 0  # number of words

        self.z_m = []  # the topic assignment for each document
        self.m_z = np.zeros([])  # the number of documents under per topic
        self.n_zq_t = np.zeros([])  
        self.n_zq = np.zeros([])  # the number of words under topic z, POS q

        self.p_z = np.zeros([])  # the probability of each topic
        self.p_m_z = np.zeros([])  # 
        self.p_m_z_sw = np.zeros([])  # 
        self.p_zq_t = np.zeros([])  # 
        self.p_t_z = np.zeros([])  # the topic distribution under words

        self.vector_path = vector_path  
        self.vector_dim = 0
        self.threshold_sim = threshold_sim
        self.filter_size = filter_size
        self.weight = weight
        self.word_vectors = defaultdict(dict)
        self.synonyms = defaultdict(dict)

        self.mh = mh
        self.alias_tables = []
        self.language = language

        self.cost_time = 0

    def initialize(self, ltp_dir, has_pre_model, t_times, t_time_ndocs, t_corpus, t_pos):
        print("has_pre_model:", has_pre_model)

        tmp_word_freq = defaultdict(int)  # word count
        if self.language == 'c':
            judge_word = judgeWordC(ltp_dir)
        elif self.language == 'e':
            # judge_word = judgeWordEBert()
            judge_word = judgeWordE()

        if has_pre_model:
            print("remove old documents...")
            ntimes_to_del = len(t_times)
            ndocs_to_del = sum(self.time_ndocs[:ntimes_to_del])
            print("ntimes_to_del:", ntimes_to_del)
            print("ndocs_to_del:", ndocs_to_del)

            for m, doc in enumerate(self.docs_word_ids[:ndocs_to_del]):
                z = self.z_m[m]
                self.m_z[z] -= 1

                for n, t in enumerate(doc):
                    self.word_freq[t] -= 1
                    q = self.docs_pos_ids[m][n]
                    if q == -1:
                        continue
                    zq = z * 3 + q if q != 3 else 3 * self.K

                    self.n_zq_t[zq, t] -= 1
                    self.n_zq[zq] -= 1

            self.times = self.times[ntimes_to_del:]
            self.time_ndocs = self.time_ndocs[ntimes_to_del:]  
            self.corpus = self.corpus[ndocs_to_del:]
            self.docs_word_ids = self.docs_word_ids[ndocs_to_del:]
            self.docs_pos_ids = self.docs_pos_ids[ndocs_to_del:]
            self.docs_occ_idxes = self.docs_occ_idxes[ndocs_to_del:]

            print("generate vocabulary...")
            for m, doc in enumerate(t_corpus):
                for n, word in enumerate(doc):
                    if judge_word.useful(word, t_pos[m][n], filter_pos=True):  # preprocessing
                        if word in self.vocas_ids:  
                            self.word_freq[self.vocas_ids[word]] += 1  # word id
                        else:  
                            tmp_word_freq[word] += 1  # word

            # Delete the words whose frequency is less than the threshold in the original corpus
            wordids_to_del = set()
            for wordid, freq in enumerate(self.word_freq):
                if freq < self.threshold_wq:
                    wordids_to_del.add(wordid)
            # 
            for word, freq in list(tmp_word_freq.items()):
                if freq < self.threshold_wq:
                    del tmp_word_freq[word]

            # Generate a new vocabulary
            old_vocas = self.vocas
            self.vocas = []
            for wordid, word in enumerate(old_vocas):
                if wordid not in wordids_to_del:
                    self.vocas.append(word)
            for word in list(tmp_word_freq.keys()):
                self.vocas.append(word)

            self.vocas_ids.clear()
            self.vocas_ids = {word: vid for vid, word in enumerate(self.vocas)}
            self.V = len(self.vocas)
            self.word_freq = [0] * self.V

            print("update old documents...")
            self.N = 0
            for m, old_doc in enumerate(self.docs_word_ids):
                word_ids, pos_ids, occ_idxes = [], [], []
                dict_occ_idx = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]

                for n, word_id in enumerate(old_doc):
                    word = old_vocas[word_id]
                    if word in self.vocas_ids:
                        q = self.docs_pos_ids[m][n]

                        word_id = self.vocas_ids[word]  # old word id -> new word id
                        word_ids.append(word_id)
                        pos_ids.append(q)
                        occ_idxes.append(dict_occ_idx[q][word_id])
                        dict_occ_idx[q][word_id] += 1
                        self.word_freq[word_id] += 1

                self.docs_word_ids[m] = word_ids
                self.docs_pos_ids[m] = pos_ids
                self.docs_occ_idxes[m] = occ_idxes
                self.N += len(word_ids)

            print("add new documents...")
            for m, t_doc in enumerate(t_corpus):
                word_ids, pos_ids, occ_idxes = [], [], []
                dict_occ_idx = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]

                for n, word in enumerate(t_doc):  # word
                    if word in self.vocas_ids:
                        q = judge_word.pos2num(t_pos[m][n])
                        if q == -1:
                            continue

                        word_id = self.vocas_ids[word]  # old word id -> new word id
                        word_ids.append(word_id)
                        pos_ids.append(q)
                        occ_idxes.append(dict_occ_idx[q][word_id])
                        dict_occ_idx[q][word_id] += 1
                        self.word_freq[word_id] += 1

                self.docs_word_ids.append(word_ids)
                self.docs_pos_ids.append(pos_ids)
                self.docs_occ_idxes.append(occ_idxes)
                self.N += len(word_ids)

            self.times.extend(t_times)
            self.time_ndocs.extend(t_time_ndocs)
            self.corpus.extend(t_corpus)
            self.M = len(self.docs_word_ids)

            print("init prior...")
            self.z_m = []  # topics of documents

            # convert the old_m_z counts to alpha priors
            self.m_z = self.m_z / sum(self.m_z) * self.K * self.alpha

            # update prev topic-word matrix 
            for word_id in sorted(wordids_to_del, reverse=True):
                self.n_zq_t = np.delete(self.n_zq_t, word_id, 1)
            smooth = np.amin(self.n_zq_t)

            num_zq = 3 * self.K + 1
            for i in range(0, len(tmp_word_freq.keys())):
                self.n_zq_t = np.append(self.n_zq_t, ([[smooth]] * num_zq), axis=1)

            # convert the pre topic-word matrix counts to proportion
            sum_n_zq_t = self.n_zq_t.sum()
            self.n_zq_t = self.n_zq_t / sum_n_zq_t * (self.V * num_zq * self.beta * self.cont) + \
                          (self.beta * (1.0 - self.cont))
            self.n_zq = self.n_zq_t.sum(axis=1)

            
        else:
            self.times = t_times
            self.time_ndocs = t_time_ndocs
            self.corpus = t_corpus

            print("generate vocabulary...")
            for m, doc in enumerate(self.corpus):
                for n, word in enumerate(doc):
                    if judge_word.useful(word, t_pos[m][n], filter_pos=True):  # preprocessing
                        tmp_word_freq[word] += 1  # word

            # 
            for word, freq in list(tmp_word_freq.items()):
                if freq < self.threshold_wq:
                    del tmp_word_freq[word]

            self.vocas = list(tmp_word_freq.keys())
            self.vocas_ids = {word: vid for vid, word in enumerate(self.vocas)}
            self.V = len(self.vocas)
            self.word_freq = [0] * self.V

            print("init documents...")
            self.M = len(self.corpus)
            self.N = 0
            for m, t_doc in enumerate(self.corpus):
                word_ids, pos_ids, occ_idxes = [], [], []
                dict_occ_idx = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]

                for n, word in enumerate(t_doc):  # word
                    if word in self.vocas_ids:  
                        q = judge_word.pos2num(t_pos[m][n])
                        if q == -1:
                            continue

                        # print(word, t_pos[m][n], q)
                        word_id = self.vocas_ids[word]
                        word_ids.append(word_id)
                        pos_ids.append(q)
                        occ_idxes.append(dict_occ_idx[q][word_id])
                        dict_occ_idx[q][word_id] += 1
                        self.word_freq[word_id] += 1

                self.docs_word_ids.append(word_ids)
                self.docs_pos_ids.append(pos_ids)
                self.docs_occ_idxes.append(occ_idxes)
                self.N += len(word_ids)

            print("init counter...")
            self.z_m = []  # topics of documents
            self.m_z = np.zeros(self.K) + self.alpha
            self.n_zq_t = np.zeros((self.K * 3 + 1, self.V)) + self.beta  
            self.n_zq = np.zeros(self.K * 3 + 1) + self.V * self.beta

        self.p_z = np.zeros(self.K)  # The probability of each topic
        self.p_zq_t = np.zeros((self.K * 3 + 1, self.V))  # The word distribution under the topic
        self.p_t_z = np.zeros((self.V, self.K))  # The topic distribution under words
        print("M:{} N:{} V:{} K:{} iters:{}".format(self.M, self.N, self.V, self.K, self.num_iters))
        print("cur_times:", self.times)
        print("cur_time_ndocs:", self.time_ndocs)

        print("init topics...")
        self.init_topics()

        print("\n-> train word vectors")
        self.train_word_vectors()
        tmp_word_freq.clear()

        if self.mh == 1:
            print("\n-> init the alias tables")
            self.init_alias_tables()

    def init_topics(self):
        """
        Random initialization
        :return:
        """
        for m, doc in enumerate(self.docs_word_ids):
            z = self.sample_doc_topic(m)
            self.z_m.append(z)
            self.m_z[z] += 1

            for n, t in enumerate(doc):
                q = self.docs_pos_ids[m][n]
                if q == -1:
                    continue
                zq = z * 3 + q if q != 3 else 3 * self.K

                self.n_zq_t[zq, t] += 1
                self.n_zq[zq] += 1

    def init_alias_tables(self):
        """
        """
        progress = ProgressBar(widgets=widgets)
        for t in progress(range(self.V)):  # 
            vose_alias = voseAlias(self.K)
            for k in range(self.K):
                zq = 3 * k + 2
                vose_alias.pw[k] = self.n_zq_t[zq, t] / (self.n_zq[zq])
                vose_alias.sum_pw += vose_alias.pw[k]

                vose_alias.construct_table()
                self.alias_tables.append(vose_alias)

    def train_word_vectors(self):
        """
        Training word2vec
        """
        self.synonyms.clear()
        print("read global word vectors...")
        word_vectors = self.read_global_word_vectors()

        print("train local word vectors...")
        word_vectors = self.train_local_word_vectors(word_vectors)

        print("get synonyms...")
        self.get_synonyms(word_vectors)

    def read_global_word_vectors(self):
        """
        read global word vectors
        :return:word_vectors
        """
        word_vectors = defaultdict(list)  # {word_id: word_vector}
        file = open(self.vector_path, "r", encoding='utf-8')
        num_global, dim_global = file.readline().split(' ')
        self.vector_dim = int(dim_global)
        print(num_global, dim_global)

        for line in file:
            items = line.strip().split(' ')
            if self.language == 'c':
                word = items[0]  # .decode('utf8')
            elif self.language == 'e':
                word = items[0].lower()
            if word not in self.vocas_ids:
                continue
            word_vectors[self.vocas_ids[word]] = np.array(list(map(eval, items[1:])))
        file.close()
        print("global_words:", len(word_vectors))
        return word_vectors

    def train_local_word_vectors(self, word_vectors):
        """
        training local word vectors
        :return:word_vectors
        """
        word2vec = Word2Vec(sentences=self.corpus, sg=1, min_count=5, size=int(self.vector_dim),  # TODO
                            workers=multiprocessing.cpu_count())
        local_words = word2vec.wv.vocab

        local_count = 0
        for word in local_words:
            if word not in self.vocas_ids:
                continue
            local_count += 1
            word_id = self.vocas_ids[word]
            vector = word2vec.wv.get_vector(word)

            if word_id in word_vectors:
                word_vectors[word_id] = (word_vectors[word_id] + vector) / 2  # TODO
            else:
                word_vectors[word_id] = vector

        for word, vec in word_vectors.items():
            word_vectors[word] = vec / np.linalg.norm(vec)

        print("local_words:", local_count)
        print("total_words:", len(word_vectors))
        self.word_vectors = word_vectors
        return word_vectors

    @exe_time
    def get_synonyms(self, word_vectors):
        """
        Calculate word similarity and get synonym dictionary
        :param word_vectors:
        :return: synonyms: synonym dictionary
        """
        wids = list(word_vectors.keys())
        num_wids = len(wids)
        distances = []

        progress = ProgressBar(widgets=widgets)
        for i in progress(range(num_wids)):
            wid_i = wids[i]
            vector_i = word_vectors[wid_i]
            for j in range(i + 1, num_wids):
                wid_j = wids[j]
                vector_j = word_vectors[wid_j]

                distance = np.dot(vector_i, vector_j)
                if distance > self.threshold_sim:
                    distances.append((wid_i, wid_j, distance))

        for word_i, word_j, distance in distances:
            self.synonyms[word_i][word_j] = distance
            self.synonyms[word_j][word_i] = distance

        # 
        for word_i, word_dist in list(self.synonyms.items()):
            if len(word_dist) == 0 or len(word_dist) > self.filter_size:
                self.synonyms.pop(word_i)

        print("synonyms:", len(self.synonyms))
        # for word_i, word_dist in self.synonyms.items():
        #     print("{}: {}".format(self.vocas[word_i], " ".join([self.vocas[i] for i in word_dist.keys()])))

    @exe_time
    def train(self, ltp_dir, has_pre_model, t_times, t_time_ndocs, t_corpus, t_pos):
        self.initialize(ltp_dir, has_pre_model, t_times, t_time_ndocs, t_corpus, t_pos)
        self.iter_training()
        self.for_evaluation()

    @exe_time
    def iter_training(self):
        print("\n-> start iter training")
        for n_iter in range(self.num_iters):
            self.update_prabability()
            print("* iterator {}".format(n_iter))
         
            for m, doc in enumerate(self.docs_word_ids):
                self.update_doc_counter(n_iter, m, -1)
                if self.mh == 0:
                    self.z_m[m] = self.sample_doc_topic(m)
                else:
                    self.z_m[m] = self.sample_doc_topic_mh(m)
                self.update_doc_counter(n_iter, m, 1)
        self.word_vectors.clear()

    def compute_likelihood(self):
        # return np.sum(np.log(self.p_m_z.sum(axis=1)[:, np.newaxis]))
        tmp = self.p_m_z.sum(axis=1)[:, np.newaxis]
        p_m = tmp[tmp != 0.0]
        return np.sum(np.log(p_m))

    def update_prabability(self):
        self.compute_p_z()  # update the probability of each topic
        self.compute_p_zq_t()  # update the word distribution under the topic
        self.compute_p_t_zq2()  # update the topic distribution under the word

    def compute_p_z(self):
        
        """
        p_topic: theta
        p_z[z] = (m_z[z] + alpha) / (sum(m_z[z] + alpha)
        :return:
        """
        self.p_z = self.m_z / sum(self.m_z)

    def compute_p_zq_t(self):
        """
        p_topic_pos_word: phi
        p_zq_t[zq][t] = (n_zq_t[zq][t] + beta) / (n_zq[t] + V * beta)
        :return:
        """
        self.p_zq_t = self.n_zq_t / self.n_zq[:, np.newaxis]

    def compute_p_t_zq2(self):
        """
        p_word_topic: topic probability given word
        p_t_z[t][z] = p_z[z] * p_z_t[z][t] / sum_k (p_z[k] * p_z_t[k][t])
        :return:
        """
        
        # 
        topic_vectors = np.zeros((self.K, self.vector_dim))
        for topic_idx in range(self.K):
            nums = 0
            for pos_idx in [0, 1, 2]:
                num_top_words = 20 if pos_idx == 2 else 5
                phi_z = self.p_zq_t[3 * topic_idx + pos_idx]

                if len(set(phi_z)) != 1:
                    for wid in phi_z.argsort()[:- num_top_words - 1:-1]:
                        if wid in self.word_vectors:
                            topic_vectors[topic_idx] += self.word_vectors[wid]
                            nums += 1
            if nums:
                topic_vec = topic_vectors[topic_idx] / nums
                topic_vectors[topic_idx] = topic_vec / np.linalg.norm(topic_vec)

        word_ids = list(self.word_vectors.keys())
        for word_id in word_ids:
            word_vector = self.word_vectors[word_id]
            for topic_idx in range(self.K):
                topic_vector = topic_vectors[topic_idx]
                distance = np.dot(word_vector, topic_vector)
                self.p_t_z[word_id, topic_idx] = distance

    def compute_p_m_z_NB(self):
        """
        Calculate the topic distribution under the document
        NB: p_m_z[m][z] = (m_z[z] + alpha) * multi_w (p_zq_t[zq][w])
        :return:
        """
        self.p_m_z = np.zeros((self.M, self.K))  
        for m, doc in enumerate(self.docs_word_ids):
            # self.p_m_z[m] = self.m_z
            self.p_m_z[m] = self.p_z
            for z in range(self.K):
                for n, t in enumerate(doc):
                    q = self.docs_pos_ids[m][n]
                    if q == -1:
                        continue
                    zq = z * 3 + q if q != 3 else 3 * self.K
                    self.p_m_z[m, z] *= self.p_zq_t[zq][t]

    def compute_p_t_z(self):
        """
        p_word_topic: topic probability given word
        p_t_z[t][z] = p_z[z] * p_z_t[z][t] / sum_k (p_z[k] * p_z_t[k][t])
        :return:
        """
        p_zq0_t = self.p_zq_t[[3 * i + 0 for i in range(self.K)]]
        p_zq1_t = self.p_zq_t[[3 * i + 1 for i in range(self.K)]]
        p_zq2_t = self.p_zq_t[[3 * i + 2 for i in range(self.K)]]
        p_zq3_t = self.p_zq_t[[3 * self.K]]

        def fuc(p_z_t, p_z):
            p_t_z = p_z_t.transpose() * p_z
            p_t_z = p_t_z / p_t_z.sum(axis=1)[:, np.newaxis]  
            return p_t_z

        return [fuc(p_zq0_t, self.p_z), fuc(p_zq1_t, self.p_z), fuc(p_zq2_t, self.p_z), fuc(p_zq3_t, self.p_z)]

    def compute_p_m_z_SW(self):
        """
        SW: p_m_z[m][z] = sum_w (p_d_w * p_w_z)
        """

        p_t_z_list = self.compute_p_t_z()
        self.p_m_z_sw = np.zeros((self.M, self.K))  # the topic distribution under the document
        for m, doc in enumerate(self.docs_word_ids):
            counts = dict(collections.Counter(self.docs_word_ids[m]))
            n_d = len(self.docs_word_ids[m])
            counts = {k: float(v) / n_d for k, v in counts.items()}

            for z in range(self.K):
                for n, t in enumerate(doc):
                    q = self.docs_pos_ids[m][n]
                    if q == -1:  
                        continue

                    if q == 3:
                        self.p_m_z_sw[m][z] += counts[t] * p_t_z_list[q][t][0]
                    else:
                        self.p_m_z_sw[m][z] += counts[t] * p_t_z_list[q][t][z]

    def update_doc_counter(self, n_iter, m, flag):
        z = self.z_m[m]
        self.m_z[z] += flag  # Number of documents under per topic

        dict_occ_idx_q2 = defaultdict(int)
        dict_occ_idx_q3 = defaultdict(int)

        for n, t in enumerate(self.docs_word_ids[m]):
            q = self.docs_pos_ids[m][n]
            if q == -1:
                continue

            if q == 0 or q == 1:  # Named entities of locations and orgnizations
                zq = z * 3 + q

            else:  # 
                if flag > 0:  # update the GPU flag when q = 2, 3. Recount the doc_word_occ_idx
                    s = self.update_gpu_flag(t, z)
                    q = 2 if s == 1 else 3
                    self.docs_pos_ids[m][n] = q

                    if q == 2:
                        self.docs_occ_idxes[m][n] = dict_occ_idx_q2[t]
                        dict_occ_idx_q2[t] += 1
                    else:
                        self.docs_occ_idxes[m][n] = dict_occ_idx_q3[t]
                        dict_occ_idx_q3[t] += 1

                if q == 2:  # apply gpu model
                    zq = z * 3 + q
                    if n_iter > 0 or flag > 0:
                        if t in self.synonyms:
                            for sim_wid, sim_dis in self.synonyms[t].items():
                                # 
                                self.n_zq_t[zq][sim_wid] += flag * sim_dis
                                self.n_zq[zq] += flag * sim_dis

                elif q == 3:
                    zq = self.K * 3

            self.n_zq_t[zq, t] += flag
            self.n_zq[zq] += flag

    def update_gpu_flag(self, t, z):
        """
        update gpu flag which decide whether do GPU operation or not according to Bernoulli(lambda_t)
        lambda_t = p(z|t)/ max p(z|t)
        :param m:
        :param z:
        :return:
        """
       
        return self.p_t_z[t, z] > 0.3

    def sample_doc_topic(self, m):
        """
        sample topic for doc according to p(z|other)
        p(zd|other) = (m_z[z] + alpha) / (D - 1 + K * alpha)
                * multi_q(0,1,2) {
                        multi_i(1 ~ N_zq) { ( n_zq_t[zq][t] + beta + docs_occ_idxes[m][w] - 1 )
                                            / (n_zq[zq] + V * beta + i - 1)
                                          }
                                 }
        :param m:
        :return:
        """
        pz = self.m_z / (self.M - 1 + self.K * self.alpha)
        pz = np.asarray(pz).astype('float64')
        # qz = pz
        for z in range(self.K):
            idx = [0, 0, 0, 0]  # 
            for n, t in enumerate(self.docs_word_ids[m]):
                q = self.docs_pos_ids[m][n]
                if q == -1:
                    continue

                zq = z * 3 + q if q != 3 else 3 * self.K
                pz[z] *= (self.n_zq_t[zq, t] + self.docs_occ_idxes[m][n]) / (self.n_zq[zq] + idx[q])
                idx[q] += 1

        if pz.sum() == 0:
            z = np.random.multinomial(1, pz).argmax()
        else:
            pz = pz / pz.sum()
            z = np.random.multinomial(1, pz).argmax()
        return z

    def sample_doc_topic_mh(self, m):
        mh_steps = 1

        z = self.z_m[m]
        old_z, new_z = z, 0
        sum_pk = self.M + self.alpha
        for r in range(mh_steps):
            # Draw a topic from corpus-proposal
            u = np.random.uniform(0, 1) * sum_pk
            if u < self.M:
                # draw from doc-topic distribution skipping n
                pos = int(u)
                new_z = self.z_m[pos]
            else:
                # draw uniformly 
                u -= self.M
                u /= self.alpha
                new_z = int(u)

            if z != new_z:
                # compute acceptance probability
                tmp_old = self.compute_pk(m, z)
                tmp_new = self.compute_pk(z, new_z)
                prop_old = self.m_z[z] + 1 + self.alpha if z == old_z else self.m_z[z] + self.alpha
                prop_new = self.m_z[new_z] + 1 + self.alpha if new_z == old_z else self.m_z[new_z] + self.alpha
                acceptance = (tmp_new * prop_old) / (tmp_old * prop_new)
                if np.random.uniform(0, 1) < acceptance:
                    z = new_z

            # Draw a topic from word-proposal
            for n, t in enumerate(self.docs_word_ids[m]):
                q = self.docs_pos_ids[m][n]  
                if q == -1 or q == 3:
                    # if q != 2:
                    continue
                zq = z * 3 + q if q != 3 else 3 * self.K

                self.alias_tables[t].num_samples += 1

                if self.alias_tables[t].num_samples > self.K:  
                    vose_alias = self.alias_tables[t]
                    vose_alias.sum_pw = 0.0
                    for k in range(self.K):
                        vose_alias.pw[k] = self.n_zq_t[zq, t] / (self.n_zq[zq])
                        vose_alias.sum_pw += vose_alias.pw[k]

                        vose_alias.construct_table()
                        self.alias_tables[t] = vose_alias

                new_z = self.alias_tables[t].sample(randint(0, self.K - 1), np.random.uniform(0, 1))

                if z != new_z:
                    # compute acceptance probability
                    tmp_old = self.compute_pk(m, z)
                    tmp_new = self.compute_pk(m, new_z)
                    acceptance = (tmp_new * self.alias_tables[t].pw[z]) \
                                 / (tmp_old * self.alias_tables[t].pw[new_z])
                    if np.random.uniform(0, 1) < acceptance:
                        z = new_z
        return z

    def compute_pk(self, m, z):
        pz = self.m_z[z] / (self.M - 1 + self.K * self.alpha)
        idx = [0, 0, 0, 0]  # 
        for n, t in enumerate(self.docs_word_ids[m]):
            q = self.docs_pos_ids[m][n]
            if q == -1:
                continue

            zq = z * 3 + q if q != 3 else 3 * self.K
            pz *= (self.n_zq_t[zq, t] + self.docs_occ_idxes[m][n]) / (self.n_zq[zq] + idx[q])
            idx[q] += 1
        return pz

    def for_evaluation(self):
        self.compute_p_z()
        self.compute_p_zq_t()
        self.compute_p_m_z_SW()
