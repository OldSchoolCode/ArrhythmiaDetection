# library setup
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import wfdb
import wget
import zipfile
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random as rand
import joblib
import pickle

database_filename = "mit-bih-arrhythmia-database-1.0.0.zip"
database_path = Path(database_filename)

if not database_path.exists():
    # this url returns a HTTP 400 error, 
    #url = f'https://storage.googleapis.com/mitdb-1.0.0.physionet.org/{database_filename}'

    # so replaced with this, which works by the look of it, and loads data into the colabs environment
    url = f'https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip'
    wget.download(url)
    with zipfile.ZipFile(database_filename, 'r') as zip_ref:
        zip_ref.extractall(".")


# RECORDS is a text file with one line for each numeric patient id
records = np.loadtxt("mit-bih-arrhythmia-database-1.0.0/RECORDS", dtype=int)

# extract the data elements we want to work with
subject_list = []

for subject in records:
    # return pointers to these objects
    record = wfdb.rdrecord(f'mit-bih-arrhythmia-database-1.0.0/{subject}')

    # subject_map is a list of variables for each patient
    subject_list.append({
        "subject": subject,
        "comment": record.comments,
        "p_signal": record.p_signal,
        "sig_len": record.sig_len,
        "sig_name": record.sig_name,
    })

print('Done')

# extract the first sequence of signals for the first patient as a list
# we know this is an MLII reading
p100_MLII = []
for a in (subject_list[0]['p_signal']):
    p100_MLII.append(a[0])

# convert from list to a np array
p100_MLII = np.array(p100_MLII)

# check this looks sensible by looking for the number of times the signal
# goes over a value of 0.55 which is roughly the R section of each heartbeat
# for an MLII reading - we know the collection time span is 30 minutes
count_beats = 0
for i in range(0, len(p100_MLII) - 1):
  if (p100_MLII[i] > 0.55) & (p100_MLII[i-1] < 0.55):
    count_beats = count_beats + 1

# check what p100_MLII looks like
print("""
len of p100_MLII {}, 
type is {}, 
snapshot{},
max value {},
min value {},
guess no of heartbeats {},
assume 70 per minute, gives {:.1f} minutes""".format(len(p100_MLII),
                                                     type(p100_MLII),
                                                     p100_MLII[0:3],
                                                     p100_MLII.max(),
                                                     p100_MLII.min(),
                                                     count_beats,
                                                     (count_beats / 70)))

# plot the array to check it look like what we're expecting
timestamps = 13000

for x in [0, 1, 2, 9, 10]:
  start = timestamps * x
  plt.title('dataset {} from {} to {}'.format(x, start, timestamps+start))
  plt.plot(np.arange(timestamps), p100_MLII[start:timestamps+start])
  plt.show()

  # Normalize the data
train_P100 = p100_MLII[0:12000]
live_P100 = p100_MLII[12001:]
training_value = (train_P100 - train_P100.min()) / \
    (train_P100.max() - train_P100.min())
training_value = training_value.reshape(len(training_value), -1)
print("Number of training samples {}, shape {}, max {} min {}".format(len(
    training_value), training_value.shape, training_value.max(), training_value.min()))

#format the data into a 3d array

# function to create each slice of teh training data
lookback = 288  # default value

# Generated training sequences for use in the model.


def create_sequences(values, lookback_datapoints=lookback):
    output = []
    for i in range(len(values) - lookback_datapoints + 1):
        output.append(values[i: (i + lookback_datapoints)])

        if (len(output) < 3):
          print(len(output), type(output))

    return np.stack(output)


x_train = create_sequences(training_value, 300)
print('training array shape is {}'.format(x_train.shape))

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
# train on the training data
history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, mode="min")
    ],
)

# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.show()

# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

# plt.hist(train_mae_loss, bins=50)
# plt.xlabel("Train MAE loss")
# plt.ylabel("No of samples")
# plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

#pickl = {'model': model}
# pickle.dump(pickl, open('model_pickle' + ".p", "wb"))
#joblib.dump(pickl, 'model.pkl')

# #Save the model
# # serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("model.h5")
print("Saved model to disk")


datapoints = []
# generate random numbers between 0-300
for d in range(301):
  value = rand.randint(0, 301)
  datapoints.append(value)

incoming_data_list = []
sequence_loss = []

for loopcount, i in enumerate(range(0, len(datapoints) - 1)):

  incoming_data_list.append(datapoints[i])

  if len(incoming_data_list) > 300:
    incoming_data_list.pop(0)

  # can't do anything until there we have the first 300 data points
  if loopcount < 300:
    continue

  if loopcount > 302:
    break

  # train the model and calculate the annomaly value
  incoming_data_array = np.array(incoming_data_list)
  training_value = (incoming_data_array - train_P100.min()) / \
      (train_P100.max() - train_P100.min())
  training_value = training_value.reshape(len(training_value), -1)

#   # create a single sequence for the model
  xin_train = create_sequences(training_value, 300)
  xin_pred = model.predict(xin_train)
  prediction = np.abs(xin_pred - xin_train)

  print(prediction)

print('Finished')
