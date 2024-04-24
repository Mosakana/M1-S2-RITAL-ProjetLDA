import numpy as np
import pickle as pkl
from preprocess import preprocess
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
import itertools
from collections import Counter

N_TOPIC = 10
EPOCHS = 1000
ALPHA = 0.5
BETA = 0.01
BATCH = 50

np.random.seed(42)


def load_docs_in_bow(path, n_doc=None):
    if n_doc:
        save_path = path + f"_{n_doc}docs.pkl"
    else:
        save_path = path + '.pkl'

    if not os.path.exists(save_path):
        with open(path, 'r', encoding='utf-8') as file:
            doc = file.read()

        docs = doc.split('\n')[:-1]

        counter_doc = 0
        indice = 0
        length = len(docs)

        if n_doc:
            doc_id = []
            collection = []
            while indice < length and counter_doc < n_doc:
                doc_id.append(docs[indice])

                doc_tmps = ''
                indice += 1
                while docs[indice] != '----- end document -----':
                    doc_tmps += preprocess(docs[indice]) + ' '
                    indice += 1

                collection.append(doc_tmps)
                counter_doc += 1
                indice += 1

        else:
            doc_id = []
            collection = []
            while indice < length:
                doc_id.append(docs[indice])

                doc_tmps = ''
                indice += 1
                while docs[indice] != '----- end document -----':
                    doc_tmps += preprocess(docs[indice]) + ' '
                    indice += 1

                collection.append(doc_tmps)
                indice += 1

        prepro = lambda x: preprocess(x)
        english_stopwords = stopwords.words('english')
        bow = CountVectorizer(preprocessor=prepro, stop_words=english_stopwords)

        vectors = bow.fit_transform(collection)
        vocabulary = bow.get_feature_names_out()
        saved_objects = [vocabulary, vectors, doc_id]

        with open(save_path, 'wb') as file:
            pkl.dump(saved_objects, file)
    else:
        with open(save_path, 'rb') as file:
            vocabulary, vectors, doc_id = pkl.load(file)

    return vocabulary, vectors, doc_id


def process_vectors(vectors):
    dict_document = {}
    n_document = vectors.shape[0]
    for d in range(n_document):
        selected_vector = vectors[d].toarray()[0]
        effective_indice = list(np.nonzero(selected_vector)[0])
        dict_document[d] = effective_indice

    return dict_document


def assign_topic(dict_document, n_topics):
    topic_assignment = {}
    for document, sequence in dict_document.items():
        topics = np.random.choice(n_topics, len(sequence), replace=True, )
        topic_assignment[document] = list(zip(sequence, topics))

    return topic_assignment


def compute_word_topic_matrix(topic_assignment, n_topics, n_vocabularies):
    Mwt = np.zeros((n_vocabularies, n_topics), dtype=np.int32)
    word_topic_counter = Counter(list(itertools.chain(*topic_assignment.values())))
    array_word_topic = np.array(list(word_topic_counter.keys()), dtype=np.int32)
    index_words = array_word_topic[:, 0]
    index_topic = array_word_topic[:, 1]
    counts = list(word_topic_counter.values())
    Mwt[index_words, index_topic] = counts

    return Mwt


def compute_document_topic_matrix(topic_assignment, n_topics):
    Mdt = np.zeros((len(topic_assignment), n_topics), dtype=np.int32)
    for document, word_topic in topic_assignment.items():
        document_topic_counter = Counter(list(np.array(word_topic, dtype=np.int32)[:, 1]))
        index_topic = list(document_topic_counter.keys())
        counts = list(document_topic_counter.values())
        Mdt[document, index_topic] = counts
    return Mdt


def get_distribution(term, document, topic_assignment, alpha, beta, n_topics, n_vocabularies):
    Mwt = compute_word_topic_matrix(topic_assignment, n_topics, n_vocabularies)
    Mdt = compute_document_topic_matrix(topic_assignment, n_topics)

    distribution = []
    for i in range(n_topics):
        Pwt = (Mwt[term, i] + beta) / (np.sum(Mwt[:, i]) + n_vocabularies * beta)
        Pdt = (Mdt[document, i] + alpha) / (np.sum(Mdt[:, i]) + n_topics * alpha)
        distribution.append(Pwt * Pdt)

    return distribution


def sampled_topic(word, document, topic_assignment, alpha, beta, n_topics, n_vocabularies):
    # normalization
    distribution = np.array(get_distribution(word, document, topic_assignment, alpha, beta, n_topics, n_vocabularies))
    distribution /= distribution.sum()

    sample_multinomial = np.random.multinomial(1, distribution)
    return np.where(sample_multinomial == 1)[0][0]


def gibbs_sampling(topic_assignment, alpha, beta, n_topics, n_vocabularies):
    new_topic_assignment = {}
    for document, word_topic in topic_assignment.items():
        new_topic_assignment[document] = []
        for word, topic in word_topic:
            new_topic_assignment[document].append(
                (word, sampled_topic(word, document, topic_assignment, alpha, beta, n_topics, n_vocabularies)))

    return new_topic_assignment


def lda(topic_assignment, n_topics, n_vocabularies, epochs, alpha, beta):
    converge = False

    for i in range(epochs):
        print(f"------- Starting {i + 1} iteration -------\n")
        last_assignment = topic_assignment
        topic_assignment = gibbs_sampling(topic_assignment, alpha, beta, n_topics, n_vocabularies)

    for res_word_topic, last_word_topic in zip(topic_assignment.values(), last_assignment.values()):
        converge = np.all(np.array(res_word_topic) == np.array(last_word_topic))

    return topic_assignment, converge


def average_document_length(dict_document):
    total_length = 0
    document_counter = 0
    for words in dict_document.values():
        total_length += len(words)
        document_counter += 1

    return total_length / document_counter


def get_word_wise_assignment(topic_assignment, n_topics, n_vocabularies):
    matrix_word_topic = np.zeros((n_vocabularies, n_topics))
    for word_topic in topic_assignment.values():
        for word, topic in word_topic:
            matrix_word_topic[word, topic] += 1

    return matrix_word_topic


def get_word_assignment_info(topic_assignment, vocabulary, n_topics, n_vocabularies, top_topic=5, top_word=10):
    word_topic_matrix = get_word_wise_assignment(topic_assignment, n_topics, n_vocabularies)
    top_topic_index = np.argsort(word_topic_matrix.sum(axis=0))[::-1]
    topic_percentage = np.sort(word_topic_matrix.sum(axis=0) / word_topic_matrix.sum())[::-1]
    print(f'------- The top {top_topic} topics -------\n')
    print(f'It\'s top {top_word} words:')
    for t in range(top_topic):
        print(f'- topic {top_topic_index[t]} ({topic_percentage[t] * 100 :.2f}%)')
        word_assigment_vector = word_topic_matrix[:, top_topic_index[t]]
        top_word_index = np.argsort(word_assigment_vector)[::-1]
        word_percentage = np.sort(word_assigment_vector / word_assigment_vector.sum())[::-1]
        for w in range(top_word):
            print(f'\t- {vocabulary[top_word_index[w]]} ({word_percentage[w] * 100:.2f}%)')


def get_document_wise_assignment(topic_assignment, n_topics):
    n_documents = len(topic_assignment)
    matrix_document_topic = np.zeros((n_documents, n_topics))
    for document, word_topic in topic_assignment.items():
        for _, topic in word_topic:
            matrix_document_topic[document, topic] += 1

    return matrix_document_topic


def get_document_assignment_info(topic_assignment, doc_id, n_topics, n_documents=5):
    document_topic_matrix = get_document_wise_assignment(topic_assignment, n_topics)
    print('The distribution of topics to documents:')
    for d in range(n_documents):
        document_topic_vector = document_topic_matrix[d]
        top_topic_index = np.argsort(document_topic_vector)[::-1]
        topic_percentage = np.sort(document_topic_vector / document_topic_vector.sum())[::-1]
        string_topic = f'{topic_percentage[0]:.3f} * topic{top_topic_index[0]}'

        for topic, percentage in zip(top_topic_index[1:], topic_percentage[1:]):
            if percentage <= 0.0001:
                break
            else:
                string_topic += ' + '
                string_topic += f'{percentage:.3f} * topic{topic}'

        print(f'\t- {doc_id[d]} : {string_topic}')


def compute_tf(term, doc):
    return Counter(doc)[term] / len(doc)


def compute_cf(dict_document):
    collection = list(itertools.chain(*dict_document.values()))
    return Counter(collection)


def dirichlet_smooth(term, doc, cf, all_cf, mu):
    return (compute_tf(term, doc) + mu * cf / all_cf) / (len(doc) + mu)


def get_term_probability(term, doc, cf, all_cf, lda_distribution, mu, sigma):
    dirichlet = dirichlet_smooth(term, doc, cf, all_cf, mu)

    return sigma * dirichlet + (1 - sigma) * sum(lda_distribution)

def query_score(query, dict_document, topic_assignment, n_topics, n_vocabularies, alpha, beta):
    cf_counter = compute_cf(dict_document)
    all_cf = sum(list(cf_counter.values()))

    scores = []
    for doc in range(len(dict_document)):
        score = 0
        for term in query:
            distribution = get_distribution(term, doc, topic_assignment, alpha, beta, n_topics, n_vocabularies)
            score += np.log(get_term_probability(term, dict_document[doc], cf_counter[term], all_cf, distribution, 1000, 0.7))

        scores.append(score)

    return scores



# path_to_doc = "../data/docs_trec_covid"
# vocabulary, vectors, doc_id = load_docs_in_bow(path_to_doc, n_doc=10)
#
# dict_document = process_vectors(vectors)
#
# topic_assignment = assign_topic(dict_document, N_TOPIC)
#
# for i in range(EPOCHS // BATCH):
#     topic_assignment, is_converge = lda(topic_assignment, N_TOPIC, len(vocabulary), BATCH, ALPHA, BETA)
#     print(f'after {(i + 1) * BATCH} iterations, learning converges : {is_converge}\n')
#     if is_converge:
#         break
#
# cf_counter = compute_cf(dict_document)
# all_cf = 0
# for i in range(len(vocabulary)):
#     all_cf += cf_counter[i]
#
# query = ['antigen', 'statist']
# int_query = list(map(lambda x: list(vocabulary).index(x), query))
# scores = query_score(int_query, dict_document, topic_assignment, N_TOPIC, len(vocabulary), ALPHA, BETA)
# print(scores)
#
# for i in range(EPOCHS):
#     topic_assignment, is_converge = lda(topic_assignment, N_TOPIC, len(vocabulary), EPOCHS, ALPHA, BETA)
# dict_document = process_vectors(vectors)
# assignment = assign_topic(dict_document, N_TOPIC, ALPHA)
# Mwt = compute_word_topic_matrix(assignment, len(vocabulary), N_TOPIC)
# Mdt = compute_document_topic_matrix(assignment, N_TOPIC)
#
# sampled_topic(1, 2, Mwt, Mdt, alpha, beta, N_TOPIC, len(vocabulary))

# after_lda = lda(dict_document, N_TOPIC, len(vocabulary), EPOCHS, ALPHA, BETA)
#
# print(after_lda)


# print(dict_doc[list(dict_doc.keys())[0]])


