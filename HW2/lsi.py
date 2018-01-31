import logging
import sys, getopt
import os
import time
import numpy as np
from collections import defaultdict
import pickle
K_VALUE = 0
MODEL_VALUE = 0
ALPHA_VALUE = 1
BETA_VALUE = 1
def main(argv):
    print(argv)
    help_str = 'lsi.py -k <samples> -m <model>'
    try:
        opts = []
        for i in range(len(argv)):
            
            if i == len(argv)-1:
                break
            else:
                opts.append((argv[i],argv[i+1]))
    except getopt.GetoptError:
        print (help_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--help':
            print("-m 1 : lsi on bow corpus")
            print("-m 2 : lsi on shifted bow corpus")
            print("-m 3 : lsi on tfidf corpus")
            print("-m 4 : lda on bow corpus")
            print("-b beta value lda")
            print("-a alpha value lda")
        if opt == '-k':
            global K_VALUE
            K_VALUE = int(arg)
        elif opt =='-m':
            global MODEL_VALUE
            MODEL_VALUE = int(arg)
        elif opt =='-a':
            global ALPHA_VALUE
            ALPHA_VALUE=float(arg)
        elif opt =='-b':
            global BETA_VALUE
            BETA_VALUE=float(arg)
if __name__ == '__main__':
    main(sys.argv[1:])


# Helper functions for quickly storing and/or retrieving python objects
def dump(obj, name):
    with open(name + '.pickle', 'wb') as file:
        pickle.dump(obj, file)
    print(name + " is stored on disk.")


def load(name):
    with open(name + '.pickle', 'rb') as file:
        return pickle.load(file)


print(K_VALUE)
print(MODEL_VALUE)
# import logging
logging.basicConfig(format = '%(asctime)s %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p',
                    filename = 'TF-IDF.log',
                    level=logging.DEBUG)
logger = logging.getLogger('lsi')
logging.info("Wha")
fh = logging.FileHandler('TF-IDF.log', mode='a')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
logger.info("HI")
logger.debug("HELLOO")
print("LOGGED")

import collections
import io
import logging
import sys
import pyndri


def parse_topics(file_or_files,
                 max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    if not isinstance(file_or_files, list) and \
            not isinstance(file_or_files, tuple):
        if hasattr(file_or_files, '__iter__'):
            file_or_files = list(file_or_files)
        else:
            file_or_files = [file_or_files]

    for f in file_or_files:
        assert isinstance(f, io.IOBase)

        for line in f:
            assert(isinstance(line, str))

            line = line.strip()

            if not line:
                continue

            topic_id, terms = line.split(delimiter, 1)

            if topic_id in topics and (topics[topic_id] != terms):
                logging.error('Duplicate topic "%s" (%s vs. %s).',
                              topic_id,
                              topics[topic_id],
                              terms)

            topics[topic_id] = terms

            if max_topics > 0 and len(topics) >= max_topics:
                break

    return topics



def write_run(model_name, data, out_f,
              max_objects_per_query=sys.maxsize,
              skip_sorting=False):
    """
    Write a run to an output file.
    Parameters:
        - model_name: identifier of run.
        - data: dictionary mapping topic_id to object_assesments;
            object_assesments is an iterable (list or tuple) of
            (relevance, object_id) pairs.
            The object_assesments iterable is sorted by decreasing order.
        - out_f: output file stream.
        - max_objects_per_query: cut-off for number of objects per query.
    """
    for subject_id, object_assesments in data.items():
        if not object_assesments:
            logging.warning('Received empty ranking for %s; ignoring.',
                            subject_id)

            continue

        # Probe types, to make sure everything goes alright.
        # assert isinstance(object_assesments[0][0], float) or \
        #     isinstance(object_assesments[0][0], np.float32)
        assert isinstance(object_assesments[0][1], str) or \
            isinstance(object_assesments[0][1], bytes)

        if not skip_sorting:
            object_assesments = sorted(object_assesments, reverse=True)

        if max_objects_per_query < sys.maxsize:
            object_assesments = object_assesments[:max_objects_per_query]

        if isinstance(subject_id, bytes):
            subject_id = subject_id.decode('utf8')

        for rank, (relevance, object_id) in enumerate(object_assesments):
            if isinstance(object_id, bytes):
                object_id = object_id.decode('utf8')

            out_f.write(
                '{subject} Q0 {object} {rank} {relevance} '
                '{model_name}\n'.format(
                    subject=subject_id,
                    object=object_id,
                    rank=rank + 1,
                    relevance=relevance,
                    model_name=model_name))


index = pyndri.Index('index/')

logger.info("index retrieved")


num_documents = index.maximum_document() - index.document_base()

dictionary = pyndri.extract_dictionary(index)
logger.info("Dictionary retrieved")
# The following writes the run to standard output.
# In your code, you should write the runs to local
# storage in order to pass them to trec_eval.
# write_run(
#     model_name='example',
#     data={
#         'Q1': ((11.0, 'DOC1'), (0.5, 'DOC2'), (0.75, 'DOC3')),
#         'Q2': ((-0.1, 'DOC1'), (1.25, 'DOC2'), (0.0, 'DOC3')),
#     },
#     out_f=sys.stdout,
#     max_objects_per_query=1000)

try:
    queries = load('queries')
except:
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        queries = parse_topics([f_topics])
    dump(queries,'queries')

# index = pyndri.Index('index/')

# num_documents = index.maximum_document() - index.document_base()

# dictionary = pyndri.extract_dictionary(index)
try:
    tokenized_queries, query_term_ids = load("make_sets")
except:
    tokenized_queries = {
        query_id: [dictionary.translate_token(token)
                   for token in index.tokenize(query_string)
                   if dictionary.has_token(token)]
        for query_id, query_string in queries.items()}

    query_term_ids = set(
        query_term_id
        for query_term_ids in tokenized_queries.values()
        for query_term_id in query_term_ids)
    make_sets = (tokenized_queries,query_term_ids)
    dump(make_sets, 'make_sets')

try:
    inverted_index = load('inverted_index')
except:
    logger.info('Gathering statistics about', len(query_term_ids), 'terms.')
    # # inverted index creation.
    start_time = time.time()

    document_lengths = {}
    unique_terms_per_document = {}

    inverted_index = defaultdict(dict)
    collection_frequencies = defaultdict(int)

    total_terms = 0

    for int_doc_id in range(index.document_base(), index.maximum_document()):
        ext_doc_id, doc_token_ids = index.document(int_doc_id)

        document_bow = collections.Counter(
            token_id for token_id in doc_token_ids
            if token_id > 0)
        document_length = sum(document_bow.values())

        document_lengths[int_doc_id] = document_length
        total_terms += document_length

        unique_terms_per_document[int_doc_id] = len(document_bow)

        for query_term_id in query_term_ids:
            assert query_term_id is not None

            document_term_frequency = document_bow.get(query_term_id, 0)

            if document_term_frequency == 0:
                continue

            collection_frequencies[query_term_id] += document_term_frequency
            inverted_index[query_term_id][int_doc_id] = document_term_frequency
    dump(inverted_index,'inverted_index')

# avg_doc_length = total_terms / num_documents

# logger.info('Inverted index creation took', time.time() - start_time, 'seconds.')
import pyndri.compat
import pickle
import time
from gensim import models
import gensim

# Datastructures:
# tokenized_queries = query_id : [term_id, term_id, term_id]
# inverted_index = term_id: {doc_id : term frequency}
# collection_frequencies = term_id: term_frequency
# run_document structure: 1000 docs per query.
# query_id, Q0, doc_id, doc_rank, query_doc score, retrieval_func
# qrel doc structure:
# docvec = [(term_id, term_freq), ...]
# Obtain term_document matrix
# Corpus format: per doc:
# [[(term_id, term_doc_freq)]]


def bow_to_tfidf(term_id, term_freq, int_doc_id, total_inverted_index):
    # for a given term_id and term_frequency, score term to dfidf
    if len(total_inverted_index[term_id]) == 0:
        # stop words (term_id=0) are not counted in given inverted index
        # log2(0) not desirable.
        logger.info("{},{},{}".format(term_id, term_freq, int_doc_id))
        return 0
    idf = np.log2(num_documents) - np.log2(len(total_inverted_index[term_id]))
    tfidf_weight = np.log2(1 + term_freq) * idf
    return tfidf_weight





def convert_no_stop(sentence):
    result = []
    for term_id, count in sentence:
        result.append((term_id - 1, count))
    return result


def bow_doc_to_idf_doc(doc_bow, int_doc_id, total_inverted_index):
    doc_tfidf = []
    for term_id, term_frequency in doc_bow:
        # tfidf_score = tfidf(int_doc_id, term_id, None)
        tfidf_score = bow_to_tfidf(
            term_id, term_frequency, int_doc_id, total_inverted_index)
        # Store 32bit float instead of 64bit for storage reasons.
        doc_tfidf.append((term_id, np.float32(tfidf_score)))
    return doc_tfidf


def train_lda_bow_corpus(index, k,model_name):
    global ALPHA_VALUE
    global BETA_VALUE
    logger.info("Starting training  LDA on bow corpus")
    standard_alpha = standard_beta = 1/k # the standard value used by the gensim library implementation of LDA
    alpha = standard_alpha * ALPHA_VALUE
    beta = standard_beta * BETA_VALUE
    dictionary = pyndri.extract_dictionary(index)
    lda_model = None
    doc_batch = []
    for int_doc_id in range(index.document_base(), index.maximum_document()):
        ext_doc_id, doc_token_ids = index.document(int_doc_id)
        if int_doc_id % 100 == 0:
            logger.info("processing doc {}/{}".format(int_doc_id, index.maximum_document()))
        # doc2bow returns a sorted list of tuples.
        doc_bow = dictionary.doc2bow(doc_token_ids)
        # remove the stop word count from doc_bow of token_id = 0, which has to be the first
        #  since it is ordered on token_ids.
        if len(doc_bow) > 0:  # protect against empty docs
            doc_bow.pop(0)
        doc_batch.append(doc_bow)
        #logger.info("Converting to tfidf took {} seconds".format(time.time() - s))
        # Train lsi model on converted doc

    logger.info("Adding batch to lda model.")
    if not lsi_model:
        logger.info("Initializing lda model")
        s = time.time()
        lda_model = models.LdaModel(corpus=doc_batch, num_topics=k,alpha=alpha,eta=beta)
        logger.info("Model trained in {} seconds.".format(time.time() - s))
    else:
        s = time.time()
        lda_model.update(doc_batch)
        logger.info("Adding document took {} seconds".format(time.time() - s))
    dump(lda_model,model_name+'_model')
    return lda_model


def train_lsi_bow_corpus_shift(index, k,model_name):
        # Trains lsi on the bow corpus with all indexes shifter negative one value. This is to
        # truly remove the existence of stopwords before training to see if it
        # makes a difference.
    dictionary = pyndri.extract_dictionary(index)
    lsi_model = None
    doc_batch = []
    for int_doc_id in range(index.document_base(), index.maximum_document()):
        ext_doc_id, doc_token_ids = index.document(int_doc_id)
        if int_doc_id % 100 == 0:
            logger.info("processing doc {}/{}".format(int_doc_id, index.maximum_document()))
        # doc2bow returns a sorted list of tuples.
        doc_bow = dictionary.doc2bow(doc_token_ids)
        # remove the stop word count from doc_bow of token_id = 0, which has to be the first
        #  since it is ordered on token_ids.
        if len(doc_bow) > 0:  # protect against empty docs
            doc_bow.pop(0)
        doc_bow = convert_no_stop(doc_bow)
        doc_batch.append(doc_bow)
        #logger.info("Converting to tfidf took {} seconds".format(time.time() - s))
        # Train lsi model on converted doc

    logger.info("Adding batch to lsi model.")
    if not lsi_model:
        logger.info("Initializing lsi model")
        s = time.time()
        lsi_model = models.LsiModel(corpus=doc_batch, num_topics=k)
        logger.info("Model trained in {} seconds.".format(time.time() - s))
    else:
        s = time.time()
        lsi_model.add_documents(doc_batch)
        logger.info("Adding document took {} seconds".format(time.time() - s))
    dump(lsi_model,model_name+'_model')
    return lsi_model


def train_lsi_bow_corpus(index, k,model_name):
    dictionary = pyndri.extract_dictionary(index)
    lsi_model = None
    doc_batch = []
    for int_doc_id in range(index.document_base(), index.maximum_document()):
        ext_doc_id, doc_token_ids = index.document(int_doc_id)
        if int_doc_id % 100 == 0:
            logger.info("processing doc {}/{}".format(int_doc_id, index.maximum_document()))
        # doc2bow returns a sorted list of tuples.
        doc_bow = dictionary.doc2bow(doc_token_ids)
        # remove the stop word count from doc_bow of token_id = 0, which has to be the first
        #  since it is ordered on token_ids.
        if len(doc_bow) > 0:  # protect against empty docs
            doc_bow.pop(0)
        doc_batch.append(doc_bow)
        #logger.info("Converting to tfidf took {} seconds".format(time.time() - s))
        # Train lsi model on converted doc

    logger.info("Adding batch to lsi model.")
    if not lsi_model:
        logger.info("Initializing lsi model")
        s = time.time()
        lsi_model = models.LsiModel(corpus=doc_batch, num_topics=k)
        logger.info("Model trained in {} seconds.".format(time.time() - s))
    else:
        s = time.time()
        lsi_model.add_documents(doc_batch)
        logger.info("Adding document took {} seconds".format(time.time() - s))
    dump(lsi_model,model_name+'_model')
    return lsi_model


def train_lsi_tfidf_corpus(index, k,model_name):
        # This method employs online learning of lsi on all the documents in the corpus.
        # Online learning is chosen to avoid memory issues.
    logger.info("Starting training  lsi on tfidf corpus")
    dictionary = pyndri.extract_dictionary(index)
    lsi_model = None
    doc_batch = []
    for int_doc_id in range(index.document_base(), index.maximum_document()):
        # for every document in the index
        ext_doc_id, doc_token_ids = index.document(int_doc_id)
        # doc2bow returns a sorted list of tuples.
        if int_doc_id % 100 == 0:
            logger.info("processing doc {}/{}".format(int_doc_id, index.maximum_document()))
        # get bag of words representation of doc
        doc_bow = dictionary.doc2bow(doc_token_ids)
        # remove the stop word count from doc_bow of token_id = 0, which has to be the first
        #  since it is ordered on token_ids.
        if len(doc_bow) > 0:  # protect against empty docs
            doc_bow.pop(0)
        # convert to tfidf
        #s = time.time()
        doc_tfidf = bow_doc_to_idf_doc(
            doc_bow, int_doc_id, total_inverted_index)
        doc_batch.append(doc_tfidf)
        #logger.info("Converting to tfidf took {} seconds".format(time.time() - s))
        # Train lsi model on converted doc
    logger.info("Adding batch to lsi model.")
    if not lsi_model:
        logger.info("Initializing lsi model")
        s = time.time()
        lsi_model = models.LsiModel(corpus=doc_batch, num_topics=k)
        logger.info("Model trained in {} seconds.".format(time.time() - s))
    
        # s=time.time()
        # lsi_model.add_documents(doc_batch)
        #logger.info("Adding document took {} seconds".format(time.time() - s))
    dump(lsi_model,model_name+'_model')
    return lsi_model


def get_total_inverted_index():
    # in order to calculate the tfidf values for document terms that do not
    #  appear in any query we need the total inverted index that maps all
    #  document counts to all the terms (not just query terms).
    # This method is a copy of the given method constructing the
    #  inverted index, but counts all terms.
    total_inverted_index = defaultdict(dict)
    for int_doc_id in range(index.document_base(), index.maximum_document()):
        if int_doc_id % 100 == 0:
            logger.info("Processing doc id {}/{}".format(int_doc_id,
                                                   index.maximum_document()))
        ext_doc_id, doc_token_ids = index.document(int_doc_id)
        document_bow = collections.Counter(
            token_id for token_id in doc_token_ids
            if token_id > 0)
        num_tokens = len(id2token)
        for term_id in range(1, num_tokens + 1):
            # we know that term ids are incremental starting from 1.
            document_term_frequency = document_bow.get(term_id, 0)
            if document_term_frequency == 0:
                continue
            inverted_index[term_id][int_doc_id] = document_term_frequency
    return inverted_index

# s = time.time()
logger.info("Creating total inverted index")
logger.info("Loading total_inverted_index")
total_inverted_index = load('total_inverted_index')
# logger.info("Done getting total inverted index in {} seconds".format(time.time()-s))
# s = time.time()
# dump(total_inverted_index,'total_inverted_index')
# logger.info("Getting bow_corpus...")

# bow_corpus = get_bow_corpus(index)
# dump(bow_corpus,'bow_corpus')
# bow_corpus = load('bow_corpus')
# logger.info("Getting no_stop_bow_corpus...")
# no_stop_bow_corp = [convert_no_stop(sentence) for sentence in bow_corpus]
# dump(no_stop_bow_corp,'no_stop_bow_corp')


def run_lsi(model_name, lsi_train_func, lsi_score_func, k):
    """
    Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.

    :param model_name: the name of the model (a string)
    :param score_fn: the scoring function (a function - see below for an example)
    """
    logger.info("Running lsi func")
    tfidf_ranking_file = 'tfidf.run'

    run_out_path = '{}.run'.format(model_name)

    def get_int(ext_id):
        return index.document_ids([ext_id])[0][1]

    if os.path.exists(run_out_path):
        logger.info("Filename already exists:{}".format(run_out_path))
        return

    retrieval_start_time = time.time()

    logger.info('Retrieving using {}'.format(model_name))
    # Get top 1000 documents (internal ids) as ranked by tfidf for all queries
    tfidf_ranking = defaultdict(list)
    tfidf_run_file = open(tfidf_ranking_file, 'r')
    for line in tfidf_run_file.readlines():
        query_id = line.split()[0]
        ext_doc_id = line.split()[2]
        int_doc_id = get_int(ext_doc_id)
        tfidf_ranking[query_id].append((int_doc_id, ext_doc_id))

    logger.info("Training lsi model with defined training function")
    try:
        lsi_model = load(model_name+'_model')
    except:
        logger.info("No model saved yet.")
        lsi_model = lsi_train_func(index, k,model_name)
    data = {}

    # returns
    cntr = 0
    num_queries = len(tfidf_ranking.items())
    for query_id, top_document_ids in tfidf_ranking.items():
        cntr += 1
        # for each query
        query_term_ids = tokenized_queries[query_id]
        # logger.info('query', [dictionary.id2token[x] for x in query_token_list])
        query_docs_score_list = []
        logger.info("Getting ranks for query {} ({}/{})".format(query_id, cntr, num_queries))
        for int_doc_id, ext_doc_id in top_document_ids:
            # for each document in the top-1000 idf ranking for this query

            ext_doc_id2, doc_terms_ids = index.document(int_doc_id)
            assert ext_doc_id == ext_doc_id2, 'External doc ids do not match! All is lost.'
            logger.info("Scoring on index {}".format(int_doc_id))
            q_d_score = lsi_score_func(
                doc_terms_ids, query_term_ids, lsi_model, int_doc_id)

            query_docs_score_list.append((q_d_score, ext_doc_id))

        data[query_id] = query_docs_score_list

    with open(run_out_path, 'w') as f_out:
        write_run(
            model_name=model_name,
            data=data,
            out_f=f_out,
            max_objects_per_query=1000)
    logger.info("Document ranking took {} seconds".format(
        time.time() - retrieval_start_time))


def cos_sim(query, doc):
    # assume input to be of structure [(position,value),(position,value)]
    q_val = [v for k, v in query]
    d_val = [v for k, v in doc]
    num = sum(q * d for q, d in zip(q_val, d_val))
    den = np.sqrt(sum(q**2 for q in q_val)) * np.sqrt(sum(d**2 for d in d_val))

    return np.float32(np.float64(num) / np.float64(den))


def bow(tokens):
    # short hand function name
    return dictionary.doc2bow(tokens)

# lsi_fnc(doc_token_ids, query_token_ids,lsi_model)


def lsi_bow_score(doc_token_ids, query_token_ids, lsi_model, int_doc_id):
    doc_bow, query_bow = bow(doc_token_ids), bow(query_token_ids)
    doc_lsi, query_lsi = lsi_model[doc_bow], lsi_model[query_bow]
    return cos_sim(query_lsi, doc_lsi)


def lsi_bow_nostop_score(doc_token_ids, query_token_ids, lsi_model, int_doc_id):
    doc_bow, query_bow = bow(doc_token_ids), bow(query_token_ids)
    doc_bow_no_stop = convert_no_stop(doc_bow)
    query_bow_no_stop = convert_no_stop(query_bow)
    doc_lsi, query_lsi = lsi_model[
        doc_bow_no_stop], lsi_model[query_bow_no_stop]
    return cos_sim(query_lsi, doc_lsi)



def bow_pair_to_dfidf(bow_query, bow_doc, int_doc_id):
    tfidf_query = []
    tfidf_doc = []
    doc_dict = {k: v for k, v in bow_doc}

    # if tf(term_in_query;document) = 0, then tfidf for that term is 0
    # because log(1+tf(t;d)) = 0 if tf = 0
    for (term_id, term_freq) in bow_doc:
        tfidf_score = bow_to_tfidf(
            term_id, term_freq, int_doc_id, total_inverted_index)
        tfidf_doc.append((term_id, tfidf_score))
    for (term_id, term_freq) in bow_query:
        if term_id in doc_dict:
            tf = doc_dict[term_id]
            tfidf_score = bow_to_tfidf(
                term_id, tf, int_doc_id, total_inverted_index)
            tfidf_query.append((term_id, tfidf_score))
        else:
            tfidf_query.append((term_id, 0))

    return tfidf_doc, tfidf_query


def lsi_tfidf_score(doc_token_ids, query_token_ids, lsi_model, int_doc_id):
    doc_bow, query_bow = bow(doc_token_ids), bow(query_token_ids)
    doc_dfidf, query_dfidf = bow_pair_to_dfidf(query_bow, doc_bow, int_doc_id)
    doc_lsi, query_lsi = lsi_model[doc_dfidf], lsi_model[query_dfidf]
    return cos_sim(query_lsi, doc_lsi)


def lda_bow_score(doc_token_ids, query_token_ids, lda_model, int_doc_id):
    doc_bow, query_bow = bow(doc_token_ids), bow(query_token_ids)
    doc_lda, query_lda = lda_model[doc_bow], lda_model[query_bow]
    return cos_sim(query_lda, doc_lda)


# train_lsi_bow_corpus_shift --> lsi_bow_nostop_score
# train_lsi_bow_corpus --> lsi_bow_score
# train_lsi_tfidf_corpus --> lsi_tfidf_score
# train_lda_bow_corpus --> lda_bow_score
# run lsi input: model_name, lsi_train_func, lsi_score_func,
import sys

def main(args):
    help_str = 'script.py -d <difficulty> -o <outdir> -n <number> -s <box-size>'
    try:
        opts, args = getopt.getopt(argv,"hd:o:n:s:",["difficulty=", "outdir=", "number=","size="])

    except getopt.GetoptError:
        logger.info (help_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-k':
            logger.info (help_str)
            sys.exit()


logger.info("-m 1 : lsi on bow corpus")
logger.info("-m 2 : lsi on shifted bow corpus")
logger.info("-m 3 : lsi on tfidf corpus")
logger.info("-m 4 : lda on bow corpus")
logger.info("k:"+str(K_VALUE))
logger.info("m:"+str(MODEL_VALUE))


import datetime

def time_model(file_name,train_func, score_func,k):
    logger.info("Running rankings")
    s = time.time()
    logger.info(file_name)
    logger.info("Training and ranking models ha")
    run_lsi(file_name, train_func, score_func, k)
    time_delta = (time.time()-s)/60
    logger.info("Ranking took {} seconds".format(time.time() - s))
    with open('Timelogs.log','a') as f:
        time_str = str(datetime.datetime.utcnow())[:19]
        d = len("lsi_bow_shift_k@2")
        if len(file_name)>d:
            log_str = "MODEL: {} \t RUNTIME: {} mins \t SAVE_TIME {} \n".format(file_name, time_delta, time_str)
        else:
            log_str = "MODEL: {} \t \t RUNTIME: {} mins \t SAVE_TIME {} \n".format(file_name, time_delta, time_str)
        f.write(log_str)
        f.close()

def create_run_doc(model_no,k):
    
    if model_no == 3:
        time_model('lsi_tfidf_k@{}'.format(k), train_lsi_tfidf_corpus, lsi_tfidf_score, k)
    elif model_no == 2:
        time_model('lsi_bow_shift_k@{}'.format(k), train_lsi_bow_corpus_shift,lsi_bow_nostop_score, k)
    elif model_no ==1:
        time_model('lsi_bow_k@{}'.format(k), train_lsi_bow_corpus, lsi_bow_score, k)
    elif model_no == 4:
        global ALPHA_VALUE
        global BETA_VALUE
        logger.info("ALPHA: {}".format(ALPHA_VALUE))
        logger.info("BETA: {}".format(BETA_VALUE))
        model_name = 'lda_bow_k@{}a{}b{}'.format(k,ALPHA_VALUE,BETA_VALUE)
        logger.info("MODEL NAME: {}".format(model_name))
        time_model(model_name, train_lda_bow_corpus, lda_bow_score, k)
    else:
        logger.info("Model no out of bounds.")
create_run_doc(MODEL_VALUE,K_VALUE)
sys.exit()
# print('hi')
# lsi_bow_model = load('lsi_bow')
# run_lsi('lsi_bow_k@200', lsi_bow, lsi_bow_model,'tfidf.run')
