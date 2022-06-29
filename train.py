from tensorflow.keras.optimizers import *
from tensorflow.python.keras.callbacks import *

from Loss.lossyolo3 import LossYolo3
from task import TaskParser, ModelInit

if __name__ == "__main__":
    task_parser = TaskParser(r'config\pascalVoc.json')

    # 1. create generator
    train_generator, valid_generator = task_parser.create_generator()

    # Create Mode
    model = task_parser.create_model(model_init=ModelInit.random)

    boardFolder = "./TensorBoard"
    tensorboard_callback = TensorBoard(boardFolder, histogram_freq=1)

    checkpoint = ModelCheckpoint(train_generator.save_folder,
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 save_freq='epoch',
                                 verbose=1,
                                 mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=4,
                                  verbose=1,
                                  mode='min')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=8,
                                   verbose=1,
                                   restore_best_weights=True)

    csv_logger = CSVLogger('log.csv', append=False)

    call_backs = [checkpoint, csv_logger]

    model.compile(optimizer=Nadam(learning_rate=train_generator.learning_rate),
                  loss=LossYolo3(train_generator.input_size,
                                 train_generator.batch_size,
                                 train_generator.anchors_array,
                                 train_generator.pattern_shape,
                                 iou_ignore_thresh=0.5,
                                 coord_scale=1,
                                 class_scale=1,
                                 obj_scale=1,
                                 noobj_scale=1))

    model.fit(train_generator.yield_next_batch(),
              steps_per_epoch=train_generator.steps_per_epoch,
              validation_data=valid_generator.yield_next_batch(),
              validation_steps=valid_generator.steps_per_epoch,
              epochs=train_generator.epoch,
              callbacks=call_backs,
              workers=1,
              max_queue_size=8,
              initial_epoch=0)
