from bertopic import BERTopic
import pandas as pd

from utils import load_properties, clean_text

props = load_properties(filepath='ConfigFile.properties')


def get_train_data(path):
    print("[Info] Cleaning Raw text")
    raw_data = pd.read_csv(path)
    doc = raw_data.copy()
    doc['news'] = doc["Title"].map(str) + ' ' + doc["Summary"].map(str)
    doc['news'] = doc['news'].apply(lambda x: ' '.join(pd.unique(x.split())))

    def apply_fun(x):
        return clean_text(x)

    doc['cleanNews'] = doc.news.apply(apply_fun)
    doc.to_csv("Clean.csv")
    train_data = doc['cleanNews'].to_list()
    return train_data

def get_time_stamp(path):
    print("[Info] Time Stamp")
    raw_data = pd.read_csv(path)
    doc = raw_data.copy()
    time_data = doc['Published On'].to_list()
    return time_data


def bert_train_topic():
    docs = get_train_data(props['data_path'])
    print(f"[Info] Number of documents = {len(docs)}")
    topic_model = BERTopic(language="english", calculate_probabilities=True)
    print("[Info] Training Started")
    topics, _ = topic_model.fit_transform(docs)
    print("[Info] Training Done")
    topic_freq = topic_model.get_topic_freq()
    outliers = topic_freq['Count'][topic_freq['Topic'] == -1].iloc[0]
    print(f"[Info] {outliers} documents have not been classified")
    print(
        f"[Info] The other {topic_freq['Count'].sum() - outliers} documents are {topic_freq['Topic'].shape[0] - 1} topics")
    topic_model.save('topic-model')
    print('[Info] model save done')
    print('[Info] Generating Chart Data')
    topic_model.get_topic_info().to_csv('Data/predicted_topics.csv')
    fig_clustering = topic_model.visualize_hierarchy(top_n_topics=50, width=800, height=1024)
    fig_clustering.write_html("templates/Clustering.html")
    fig_top = topic_model.visualize_topics()
    fig_top.write_html("templates/TopicDistance.html")
    fig_sim = topic_model.visualize_heatmap(n_clusters=42, top_n_topics=100)
    fig_sim.write_html("templates/simMatrix.html")
    fig_tw = topic_model.visualize_barchart(top_n_topics=10, height=1024, width=1024)
    fig_tw.write_html("templates/topicWord.html")
    print('[Info] Chart Data Generated!!')

def bert_train_dynamic_topic():
    docs = get_train_data(props['data_path'])
    timestamps = get_time_stamp(props['data_path'])
    print(f"[Info] Number of documents = {len(docs)}")
    topic_model = BERTopic(language="english", calculate_probabilities=True)
    print("[Info] Training Started")
    path = props['model_to_deploy']
    print(f'[Info] Model Path - {path}')
    topics, _ = BERTopic.load(path)
    print("[Info] Training Done")
    topics_over_time = topic_model.topics_over_time(docs, topics, timestamps, nr_bins=20)
    fig_dynamic = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
    fig_dynamic.write_html("templates/dynamicTopic.html")


def bert_predict(input_text):
    path = props['model_to_deploy']
    print(f'[Info] Model Path - {path}')
    topic_model = BERTopic.load(path)
    print('[Info] Model Loaded')
    clean_text_processed = clean_text(input_text)
    bert_pred = topic_model.transform([clean_text_processed])
    topics_data = pd.read_csv(r'Data/predicted_topics.csv')
    name_of_topic = topics_data.loc[topics_data['Topic'] == bert_pred[0][0], 'Name'].tolist()
    ret_text = name_of_topic[0]
    return str(ret_text)