# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
from optparse import OptionParser
from progressbar import *
from src.model.TSSE_DMM import TSSE_DMM
from src.utils.common import _readBunch, _writeBunch, _readFile, _writeFile, docHandlerC, docHandlerE
from src.utils.timetest import *
import importlib

# from src.bert_ner_tag.bert_ner_and_tagging import process_ner_and_tagging

sys.path.append('../../')
importlib.reload(sys)
# sys.setdefaultencoding('utf8')

widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]


def read_data(has_pre_model, corpus_path, language, ltp_dir=None):
    """
    :param:
    :return:
    """
    raw_data = _readFile(corpus_path)
    nums_raw_data = len(raw_data)
    print("corpus_path:", corpus_path)
    print("corpus:", nums_raw_data)

    t_times, t_time_ndocs, t_corpus, t_pos = [], [], [], []
    if language == 'c':
        doc_hander = docHandlerC(ltp_dir)
        progress = ProgressBar(widgets=widgets)
        for i in progress(range(nums_raw_data)):
            raw_words_u, raw_postags = doc_hander.cut(raw_data[i])
            t_corpus.append(raw_words_u)
            t_pos.append(raw_postags)

    elif language == 'e':
        # t_corpus, t_pos = process_ner_and_tagging(raw_data)

        doc_hander = docHandlerE()
        progress = ProgressBar(widgets=widgets)
        for i in progress(range(nums_raw_data)):
            raw_words_u, raw_postags = doc_hander.cut(raw_data[i])
            t_corpus.append(raw_words_u)
            t_pos.append(raw_postags)

    if not has_pre_model:
        t_times = []
        t_time_ndocs = []
    else:
        t_times = [corpus_path.split('/')[-1]]
        t_time_ndocs = [len(t_corpus)]

    return t_times, t_time_ndocs, t_corpus, t_pos


def get_model_path(corpus_path, options):
    """
    :param corpus_path:
    :return:
    """
    res_dir = os.path.join(os.path.dirname(corpus_path), "result_single")
    file_name = os.path.basename(corpus_path)
    model_path = os.path.join(
        os.path.join(res_dir, "{}_k{}_iters{}".format(file_name, options.K, options.num_iters)))
    print("model path:", model_path)
    return model_path


def get_top_words(phi, vocas, num_top_words):
    """
    Get top words
    :param phi:
    :param vocas:
    :param num_top_words:
    :return:
    """
    top_words_list = []
    for idx, phi_z in enumerate(phi):
        if len(set(phi_z)) != 1:
            top_words = [vocas[i] for i in phi_z.argsort()[:-num_top_words - 1:-1]]
            top_words_list.append(" ".join(top_words))
    return top_words_list


def save_model(options):
    model_name = get_model_path(options.t_corpus_path, options)

    model_path = model_name + '.dat'
    _writeBunch(model_path, model)

    # save top_words
    phi = []
    for idx, phi_zq in enumerate(model.p_zq_t):
        if idx % 3 == 2:
            phi.append(phi_zq)
    vocas = model.vocas

    for t in [10]:
        top_words_list = get_top_words(phi, vocas, num_top_words=t)
        top_words_path = model_name + "_top" + str(t) + ".topWords"
        print("saving top words to {}".format(top_words_path))
        _writeFile(top_words_path, top_words_list)

    # save p_m_z_SW
    res = map(lambda x: " ".join(map(lambda y: str(y), x)), model.p_m_z_sw)
    theta_path = model_name + "_SW.theta"
    print("saving p_m_z_SW to {}".format(theta_path))
    _writeFile(theta_path, res)


if __name__ == "__main__":
    print("======================= start =========================")
    parser = OptionParser()
    parser.add_option("--alpha", dest="alpha", type="float", default=0.001)
    parser.add_option("--beta", dest="beta", type="float", default=0.01)
    parser.add_option("--K", dest="K", type="int", default=40)
    parser.add_option("--num_iters", dest="num_iters", type="int", default=100)
    parser.add_option("--threshold_wq", dest="threshold_wq", type="int", default=10)
    parser.add_option("--cont", dest="cont", type="float", default=0.5)
    parser.add_option("--num_top_words", dest="num_top_words", type="int", default=20)
    parser.add_option("--has_pre_model", dest="has_pre_model", type="int", default=0)
    parser.add_option("--pre_corpus_path", dest="pre_corpus_path", type="string",
                      default="../../corpus_snippets/docs")
                      # default="../../corpus_sogouCA/docs")
    parser.add_option("--t_corpus_path", dest="t_corpus_path", type="string",
                      default="../../corpus_snippets/docs")
                      # default = "../../corpus_sogouCA/docs")
    parser.add_option("--ltp_dir", dest="ltp_dir", type="string",
                      default="../../ltp_data_v3.4.0")
    parser.add_option("--vector_path", dest="vector_path", type="string",
                      default="../../word2vec/merge_sgns_bigram_char300.txt")
    parser.add_option("--threshold_sim", dest="threshold_sim", type="float", default=0.5)
    parser.add_option("--filter_size", dest="filter_size", type="int", default=20)
    parser.add_option("--weight", dest="weight", type="float", default=0.1)
    parser.add_option("--mh", dest="mh", type="int", default=0)
    parser.add_option("--language", dest="language", type="string", default='e')

    (options, args) = parser.parse_args()
    if options.has_pre_model is None or options.pre_corpus_path is None or options.t_corpus_path is None:
        parser.error("need has_pre_model, pre_corpus_path, t_corpus_path")

    print("\n-> read training data")
    t_times, t_time_ndocs, t_corpus, t_pos = read_data(options.has_pre_model,
                                                       options.t_corpus_path,
                                                       options.language,
                                                       options.ltp_dir)

    print("\n-> train topic model")
    if options.language == 'e':
        options.vector_path = '../../word2vec/wiki-news-300d-1M.vec'
    if options.has_pre_model == 0:
        model = TSSE_DMM(options.alpha, options.beta, options.K, options.num_iters, options.threshold_wq, options.cont,
                         options.vector_path, options.threshold_sim, options.filter_size, options.weight, options.mh,
                         options.language)
    else:
        pre_model_path = get_model_path(options.pre_corpus_path, options) + '.dat'
        model = _readBunch(pre_model_path)
        print("pre_times:", model.times)
        print("pre_time_ndocs:", model.time_ndocs)

    model.train(options.ltp_dir, options.has_pre_model, t_times, t_time_ndocs, t_corpus, t_pos)

    print("\n-> save model")
    save_model(options)
