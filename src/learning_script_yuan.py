from lda import lda
from lda import load_docs_in_bow, process_vectors, assign_topic
import pickle as pkl


EPOCHS = 50
# ALPHA = 50 / N_TOPICS
BETA = 0.01
list_n_topics = [300, 400, 500]

path_to_doc = "../data/docs_trec_covid_selected_docs"
vocabulary, vectors, doc_id = load_docs_in_bow(path_to_doc)
dict_document = process_vectors(vectors)

save_path = '../data/learning_data_covid'
for n_topics in list_n_topics:
    print(f"processing {n_topics} topics...")
    output_path = f"{save_path}_{n_topics}_topics.pkl"
    ALPHA = 50 / n_topics
    topic_assignment = assign_topic(dict_document, n_topics)
    topic_assignment = lda(topic_assignment, n_topics, len(vocabulary), EPOCHS, ALPHA, BETA)
    with open(output_path, "wb") as file:
        pkl.dump(topic_assignment, file)