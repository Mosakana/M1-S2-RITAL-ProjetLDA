import numpy as np
import pickle as pkl
from preprocess import preprocess
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
import itertools
from collections import Counter

N_TOPIC = 100
EPOCHS = 100
ALPHA = 50 / N_TOPIC
BETA = 0.01


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
        topics = np.random.choice(n_topics, len(sequence), replace=True)
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


def sampled_topic(word, document, Mwt, Mdt, alpha, beta, n_topics, n_vocabularies):
    distribution = []
    for i in range(n_topics):
        Pwt = (Mwt[word, i] + beta) / (np.sum(Mwt[:, i]) + n_vocabularies * beta)
        Pdt = (Mdt[document, i] + alpha) / (np.sum(Mdt[:, i]) + n_topics * alpha)
        distribution.append(Pwt * Pdt)

    # normalization
    distribution = np.array(distribution)
    distribution /= distribution.sum()

    sample_multinomial = np.random.multinomial(1, distribution)
    return np.where(sample_multinomial == 1)[0][0]


def gibbs_sampling(topic_assignment, alpha, beta, n_topics, n_vocabularies):
    Mwt = compute_word_topic_matrix(topic_assignment, n_topics, n_vocabularies)
    Mdt = compute_document_topic_matrix(topic_assignment, n_topics)

    new_topic_assignment = {}
    for document, word_topic in topic_assignment.items():
        new_topic_assignment[document] = []
        for word, topic in word_topic:
            new_topic_assignment[document].append((word, sampled_topic(word, document, Mwt, Mdt, alpha, beta, n_topics, n_vocabularies)))

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





# path_to_doc = "../data/docs_trec_covid"
# vocabulary, vectors = load_docs_in_bow(path_to_doc, n_doc=10)
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
