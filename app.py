import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
import random

from bert import bert_predict

app = Flask(__name__,
            template_folder='templates')

charts_data = pd.read_csv(r'Data/predicted_topics.csv')

labels = charts_data.Name.to_list()
values = charts_data.Count.to_list()
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(len(values))]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/bar')
def bar():
    bar_labels = labels
    bar_values = values
    return render_template('bar_chart.html', title='Topic wise News', max=400, labels=bar_labels,
                           values=bar_values)


@app.route('/line')
def line():
    line_labels = labels
    line_values = values
    return render_template('line_chart.html', title='Topic wise News', max=400, labels=line_labels,
                           values=line_values)


@app.route('/pie')
def pie():
    pie_labels = labels
    pie_values = values
    return render_template('pie_chart.html', title='Topic wise News', max=17000,
                           set=zip(values, labels, colors))


@app.route('/get_cluster_details')
def cluster_viz():
    return render_template('Clustering.html')


@app.route('/get_topic_details')
def topic_viz():
    return render_template('TopicDistance.html')


@app.route('/get_sim_matrix')
def sim_viz():
    return render_template('simMatrix.html')


@app.route('/get_top_topics')
def word_viz():
    return render_template('topicWord.html')


@app.route('/predict', methods=['POST'])
def predict():
    # # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # # prediction = model.predict(final_features)
    int_features = [str(x) for x in request.form.values()]
    raw_text = ' '.join(int_features)
    prediction_text = bert_predict(raw_text)
    return render_template('index.html', prediction_text='Topic {}'.format(prediction_text))


@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])

    output = 0
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True,
            host='0.0.0.0')
