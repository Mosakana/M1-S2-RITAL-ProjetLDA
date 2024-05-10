import numpy as np
from lda import compute_cf, compute_tf, get_distribution
from preprocess import preprocess

def get_qrels(path, doc_id):
    with open(path, 'r', encoding='utf-8') as file:
        qrels_file = file.read()
    qrels = qrels_file.split('\n')[:-1]

    dict_relevance = {}
    for indice in range(0, len(qrels), 3):
        qrel_id = int(qrels[indice])
        d_id = qrels[indice+1]
        relevance = int(qrels[indice+2])
        if d_id in doc_id:
            if qrel_id not in dict_relevance:
                dict_relevance[qrel_id] = [(doc_id.index(d_id), relevance)]
            else:
                dict_relevance[qrel_id].append((doc_id.index(d_id), relevance))

    return dict_relevance

def get_queries(path):
    with open(path, 'r', encoding='utf-8') as file:
        queries_file = file.read()
    queries = queries_file.split('\n')[:-1]

    list_queries = []
    for indice in range(0, len(queries), 4):
        text = f'{queries[indice + 1]} {queries[indice + 2]} {queries[indice + 3]}'
        list_queries.append(preprocess(text))

    return list_queries

def queries_string2int(list_queries, vocabulary):
    list_int_queries = []
    for query in list_queries:
        int_query = []
        terms = query.split(' ')
        for term in terms:
            if term in vocabulary:
                int_query.append(list(vocabulary).index(term))
        list_int_queries.append(int_query)

    return list_int_queries


def dirichlet_smooth(term, doc, cf, all_cf, mu):
    return (compute_tf(term, doc) + mu * cf / all_cf) / (len(doc) + mu)


def get_term_probability(term, doc, cf, all_cf, lda_distribution, mu, sigma):
    dirichlet = dirichlet_smooth(term, doc, cf, all_cf, mu)

    return sigma * dirichlet + (1 - sigma) * sum(lda_distribution)

def query_score(query, dict_document, Mwt, Mdt, n_topics, n_vocabularies, alpha, beta):
    cf_counter = compute_cf(dict_document)
    all_cf = sum(list(cf_counter.values()))

    scores = []
    for doc in range(len(dict_document)):
        score = 0
        for term in query:
            distribution = get_distribution(term, doc, Mwt, Mdt, alpha, beta, n_topics, n_vocabularies)
            score += np.log(get_term_probability(term, dict_document[doc], cf_counter[term], all_cf, distribution, 1000, 0.7))

        scores.append(score)

    return scores


