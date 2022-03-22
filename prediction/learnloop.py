# from sqlalchemy import true
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#set up generic model
def build_and_compile_model(norm):
  model = tf.keras.models.Sequential( [
    norm,
    tf.keras.layers.Dense(64, activation='relu'), #input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
  return model

#load data
harmonics_raw = pd.read_csv("/home/ben/Documents/Documents/work/FARS/StandardShunt/octave/trimheaders.csv")

#insert 0% THD samples every 500 samples
harmonics_zeros = (harmonics_raw.iloc[499::500]
                   .rename(lambda x: x + .5))
harmonics_zeros.iloc[0:harmonics_raw.columns.size] = 0
harmonics_zeros['h1'] = harmonics_raw['h1'].mean()
harmonics_train = pd.concat([harmonics_raw, harmonics_zeros], sort=False).sort_index().reset_index(drop=True)

#split into training and testing data
harmonics_test = harmonics_train.sample(frac=0.25)
harmonics_train = harmonics_train.drop(harmonics_test.index)
harmonics_train_features = harmonics_train.copy()
harmonics_test_features = harmonics_test.copy()

#build normalizer to put all the values into the same range but in proportion
thda = np.array(harmonics_train_features['THD average'])
thda_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
thda_normalizer.adapt(thda)

#prepare output frame
columnnames = [ 'thd', 'frequency' ]
for n in range(1, 51, 1):
    columnnames.append('h' + str(n) )
thds = np.linspace(0, 50, 101)
outputframe = pd.DataFrame(columns=columnnames)
outputframe['thd'] = thds
outputframe['frequency'] = 50.0
outputframe['h1'] = harmonics_raw['h1'].mean()

for n in range(2, 51, 1):
  channel='h' + str(n)
  #split labels out from features
  train_labels = harmonics_train_features.pop(channel)
  test_labels = harmonics_test_features.pop(channel)

  #train model
  model = build_and_compile_model(thda_normalizer)
  model.build()
  model.fit(
      harmonics_train_features['THD average'],
      train_labels,
      epochs=10,
      validation_split = 0.2)
  #make predictions
  predictions = model.predict(thds)
  outputframe[channel]=predictions

outputframe.clip(lower=0, inplace=True)
outputframe.to_csv('predictions.csv', index=False)
