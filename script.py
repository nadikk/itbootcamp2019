import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use("classic")
from keras.models import load_model
from matplotlib.lines import Line2D
from collections import defaultdict

timesteps=200
classes = ['trapper ned', 'løbe', 'sidde', 'stå', 'trapper op', 'gå']

def load_data(path,timesteps=timesteps):
    data = pd.read_csv(path)
    to_drop = ["timestamp",
                "timeIntervalSince1970",
                "magneticField.x",
                "magneticField.y",
                "magneticField.z",
                "magneticField.accuracy"]
    data.drop(to_drop,axis=1,inplace=True)
    if data.shape[0] < 200:
        print("Ikke nok data")
    chunks = [data[i:i+timesteps] for i in range(0,data.shape[0],timesteps)]
    if len(chunks[0]) != len(chunks[-1]):
        chunks = chunks[:-1]
    concat = pd.concat(chunks,ignore_index=True)
    return concat

def load_models(kfold=5):
    models = []
    for i in range(kfold):
        models.append(load_model('itbootcamp2019/models/LSTM_{}_7'.format(i)))
    return models

def predictions(data,models,timesteps=timesteps):
    X = data.values.reshape((-1,timesteps,12))
    y_hat = np.zeros((X.shape[0],6))
    for model in models:
        y_hat += model.predict(X)
    y_hat = np.argmax(y_hat,axis=1)
    preds = []
    for i in y_hat:
        lst = [i for j in range(timesteps)]
        preds.extend(lst)
    return preds

def plot(data,preds,colors=["blue","black","red","cyan","magenta","green"]):
    # divide into segments
    data_with_labels = data.copy()
    data_with_labels["labels"] = preds
    segments = defaultdict(list)
    segment_num = 0
    for idx, row in data_with_labels.iterrows():
        if idx == 0:
            segments[(segment_num,row['labels'])].append(idx)
        elif idx == len(data_with_labels)-1 and data_with_labels['labels'][idx] == data_with_labels['labels'][idx-1]:
            segments[(segment_num,row['labels'])].append(idx)
        elif idx == len(data_with_labels)-1 and data_with_labels['labels'][idx] != data_with_labels['labels'][idx-1]:
            segment_num += 1
            segments[(segment_num,row['labels'])].append(idx)
        elif row['labels'] == data_with_labels['labels'][idx-1]:
            segments[(segment_num,row['labels'])].append(idx)
        else:
            segment_num += 1
            segments[(segment_num,row['labels'])].append(idx)
    # plot
    fig = plt.figure(figsize=(10,30))
    for idx,col in enumerate(data.columns):
        ax = fig.add_subplot(12,1,idx+1)
        for key,idxs in segments.items():
            data[col][idxs].plot(ax=ax,c=colors[int(key[1])])
    
        custom_lines = [
            Line2D([0],[0],color=colors[i]) for i in range(6)
        ]
    
        ax.legend(custom_lines,classes,loc=(1.2,0.5))
        ax.set_title(col)
    plt.show()

#data = load_data("../tests/wlksit/DeviceMotion.csv")
#models = load_models()
#preds = predictions(data,models)
#plot(data,preds)

