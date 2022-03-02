from tensorflow.python.keras.callbacks import *
from tensorflow.keras.optimizers import *

from Config.train_config import *
from DataSet.batch_generator import BatchGenerator
from Loss.loss import Loss
from Nets.yolo3_net import get_yolo3_backend
from task import TaskParser, ModelInit

if __name__ == "__main__":
    task_parser = TaskParser(r'config\raccoon.json')

    # 1. create generator
    train_generator, valid_generator = task_parser.create_generator()

    # Create Mode
    model = task_parser.create_model(model_init=ModelInit.random)

    weight_file = os.path.join(train_generator.save_folder, 'ep{epoch:03d}-loss{loss:.3f}-val-loss{val_loss:.3f}.h5')
    checkpoint = ModelCheckpoint(weight_file,
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 save_freq='epoch',
                                 verbose=1,
                                 mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  verbose=1,
                                  mode='min')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=25,
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
