from tensorflow.python.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

from Loss.loss import Loss
from task import TaskParser, ModelInit

if __name__ == "__main__":
    task_parser = TaskParser(r'config\raccoon.json')

    # 1. create generator
    train_generator, valid_generator = task_parser.create_generator()

    # Create Mode
    model = task_parser.create_model(model_init=ModelInit.random)

    checkpoint = ModelCheckpoint(train_generator.save_folder,
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 save_freq='epoch',
                                 verbose=1,
                                 mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=3,
                                  verbose=1,
                                  mode='min')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=10,
                                   verbose=1,
                                   restore_best_weights=True)

    csv_logger = CSVLogger('log.csv', append=False)

    current_net_size = (train_generator.input_size, train_generator.input_size)

    loss_bin = Loss(current_net_size,
                    train_generator.batch_size,
                    train_generator.anchors_array,
                    train_generator.pattern_shape)

    model.compile(optimizer=RMSprop(learning_rate=train_generator.learning_rate),
                  loss=loss_bin)

    model.fit(train_generator.get_next_batch(),
              steps_per_epoch=train_generator.steps_per_epoch,
              validation_data=valid_generator.get_next_batch(),
              validation_steps=valid_generator.steps_per_epoch,
              epochs=100,
              callbacks=[checkpoint, reduce_lr, early_stopping, csv_logger],
              initial_epoch=0)


