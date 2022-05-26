import random
import tensorflow as tf
from matplotlib import pyplot


def train_test_split_timeseries(df_all, ratio):
  random.seed(500)
  ids = list(df_all.project_id.unique())
  ids = random.sample(ids, int(len(ids) * ratio ) )

  return df_all[(df_all.project_id.isin(ids))], df_all[~(df_all.project_id.isin(ids))]

from matplotlib import pyplot

def create_checkpoint(name):
  checkpoint_filepath = './sample_data/weights_'+name+'.{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
  return model_checkpoint_callback

# ['accuracy', 'loss']
def show_stats(history, stat_name, name_to_store = None):
  if stat_name not in history.history.keys():
    raise ValueError('metric for model not defined')

  pyplot.plot(history.history[stat_name])
  pyplot.plot(history.history['val_'+stat_name])
  pyplot.title('model '+ stat_name)
  pyplot.ylabel(stat_name)
  pyplot.xlabel('epoch')
  pyplot.legend(['train', 'validation'], loc='upper left')
    
  pyplot.show()

def train_model(model ,train_generator, num_epochs, batch_size, valid_generator, model_checkpoint):
  callbacks = [] if model_checkpoint is None else [model_checkpoint]
  history =  model.fit(train_generator, epochs=num_epochs, batch_size=batch_size, validation_data=valid_generator, callbacks=callbacks)
  return history