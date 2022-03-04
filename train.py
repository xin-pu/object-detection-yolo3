import shutil

from tensorflow.python.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

from Loss.loss import Loss
from task import TaskParser, ModelInit

if __name__ == "__main__":
    task_parser = TaskParser(r'config\pascal_voc.json')

    # 1. create generator
    train_generator, valid_generator = task_parser.create_generator()

    # Create Mode
    model = task_parser.create_model(model_init=ModelInit.random)

    boardFolder = "./TensorBoard"
    tensorboard_callback = TensorBoard(boardFolder, histogram_freq=1)

    checkpoint = ModelCheckpoint(train_generator.save_folder,
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 save_freq='epoch',
                                 verbose=1,
                                 mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.5,
                                  patience=5,
                                  verbose=1,
                                  mode='min')

    early_stopping = EarlyStopping(monitor='loss',
                                   min_delta=0,
                                   patience=10,
                                   verbose=1,
                                   restore_best_weights=True)

    csv_logger = CSVLogger('log.csv', append=False)

    current_net_size = (train_generator.input_size, train_generator.input_size)

    loss_bin = Loss(current_net_size,
                    train_generator.batch_size,
                    train_generator.anchors_array,
                    train_generator.pattern_shape,
                    class_scale=0 if train_generator.classes == 1 else 1)

    model.compile(optimizer=Adam(learning_rate=train_generator.learning_rate, clipnorm=0.001),
                  loss=loss_bin)

    model.fit(train_generator.get_next_batch(),
              steps_per_epoch=train_generator.steps_per_epoch,
              validation_data=valid_generator.get_next_batch(),
              validation_steps=valid_generator.steps_per_epoch,
              epochs=train_generator.epoch,
              callbacks=[checkpoint, reduce_lr, early_stopping],
              workers=1,
              max_queue_size=8,
              initial_epoch=0)
