import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use("classic")
from keras.models import load_model
from matplotlib.lines import Line2D
from collections import defaultdict

timesteps = 200
window_size = int(timesteps/9)
classes = ['gå', 'hoppe', 'ligge', 'løbe', 'sidde', 'stå']

features = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 
            'gravity.x', 'gravity.y', 'gravity.z',
            'rotationRate.x', 'rotationRate.y','rotationRate.z', 
            'userAcceleration.x', 'userAcceleration.y','userAcceleration.z', 
            'totalAcceleration.x', 'totalAcceleration.y','totalAcceleration.z', 
            'magnitudeTotalAcceleration', 'magnitudeUserAcceleration',
            'magnitudeGravity', 'magnitudeRotationRate', 'magnitudeTotalVelocity',
            'angleUserGravity']

def load_data(path, timesteps=timesteps, window_size=window_size):
    df = pd.read_csv(path)
    data = []
    timestamps = []
    if len(df) > 0:
        chunks = [df[i:i+timesteps] for i in range(0, df.shape[0]-timesteps, window_size)]
        chunks = chunks[:-1]
        timestamp = [chunk.iloc[0]["timestamp"] for chunk in chunks]
        timestamps.extend(timestamp)
        data.extend(chunks)
    else:
        print("Ikke nok data")
    return pd.concat(data, ignore_index=True), timestamps

def create_features(df, features=features):
    df["totalAcceleration.x"] = df["userAcceleration.x"] + df["gravity.x"]
    df["totalAcceleration.y"] = df["userAcceleration.y"] + df["gravity.y"]
    df["totalAcceleration.z"] = df["userAcceleration.z"] + df["gravity.z"]
    df["totalVelocity.x"] = (df["totalAcceleration.x"].shift() - df["totalAcceleration.x"])/0.02
    df["totalVelocity.x"] = df["totalVelocity.x"].fillna(df.iloc[1]["totalVelocity.x"])
    df["totalVelocity.y"] = (df["totalAcceleration.y"].shift() - df["totalAcceleration.y"])/0.02
    df["totalVelocity.y"] = df["totalVelocity.y"].fillna(df.iloc[1]["totalVelocity.y"])
    df["totalVelocity.z"] = (df["totalAcceleration.z"].shift() - df["totalAcceleration.z"])/0.02
    df["totalVelocity.z"] = df["totalVelocity.z"].fillna(df.iloc[1]["totalVelocity.z"])
    df["magnitudeUserAcceleration"] = np.sqrt(df["userAcceleration.x"]**2 + df["userAcceleration.y"]**2 + df["userAcceleration.z"]**2)
    df["magnitudeTotalAcceleration"] = np.sqrt(df["totalAcceleration.x"]**2 + df["totalAcceleration.y"]**2 + df["totalAcceleration.z"]**2)
    df["magnitudeGravity"] = np.sqrt(df["gravity.x"]**2 + df["gravity.y"]**2 + df["gravity.z"]**2)
    df["magnitudeRotationRate"] = np.sqrt(df["rotationRate.x"]**2 + df["rotationRate.y"]**2 + df["rotationRate.z"]**2)
    df["magnitudeTotalVelocity"] = np.sqrt(df["totalVelocity.x"]**2 + df["totalVelocity.y"]**2 + df["totalVelocity.z"]**2)
    df["angleUserGravity"] = np.arccos((df["userAcceleration.x"]*df["gravity.x"] + 
                                        df["userAcceleration.y"]*df["gravity.y"] + 
                                        df["userAcceleration.z"]*df["gravity.z"]) / 
                                    (df["magnitudeUserAcceleration"]*df["magnitudeGravity"]))
    return df[features]

def load_models(model_num="6"):
    models = []
    for i in range(5):
        models.append(load_model('lstm_models/lstm_{}_{}'.format(model_num, i)))
    return models

def predictions(data, models, timesteps=timesteps):
    X = data.values
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    X = data.values.reshape((-1, timesteps, data.shape[1]))
    preds = []
    for model in models:
        preds.append(model.predict(X))
    preds = np.argmax(np.average(preds, axis=0), axis=1)
    #label_dict = {0: "gå", 1: "hoppe", 2: "ligge", 3: "løbe", 4: "sidde", 5: "stå"}
    #return [label_dict[y] for y in preds]
    return preds

def plot(data, preds, timesteps=timesteps, classes=classes, colors=["blue","black","red","cyan","magenta","green"]):
    # divide into segments
    data_with_labels = data.copy()
    data_with_labels["labels"] = list(np.repeat(preds, timesteps))
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
    for idx,col in enumerate(data.columns[:12]):
        ax = fig.add_subplot(12,1,idx+1)
        for key,idxs in segments.items():
            data[col][idxs].plot(ax=ax,c=colors[int(key[1])])
    
        custom_lines = [
            Line2D([0],[0],color=colors[i]) for i in range(6)
        ]
    
        ax.legend(custom_lines,classes,loc=(1.01,0.0))
        ax.set_title(col)
    plt.tight_layout()
    plt.show()

#data, timestamps = load_data("group_data/nadia_sidde_stå_gå/DeviceMotion.csv")
#print(data.shape, len(timestamps))
#data = create_features(data)
#print(data.shape)
#models = load_models()
#print(len(models))
#preds = predictions(data, models)
#print(preds)
#plot(data, preds)