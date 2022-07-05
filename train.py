from tensorflow.keras.optimizers import *
from tensorflow.python.keras.callbacks import *

from LearningScheduler import CustomLearningRateScheduler
from Loss.lossyolo3 import LossYolo3
from task import TaskParser, ModelInit

if __name__ == "__main__":
    task_parser = TaskParser(r'config\pascalVoc.json')

    # 1. create generator
    train_generator, valid_generator = task_parser.create_generator()

    # Create Mode
    model = task_parser.create_model(model_init=ModelInit.pretrain)

    checkpoint = ModelCheckpoint(train_generator.save_folder,
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 save_freq='epoch',
                                 verbose=1,
                                 mode='min')

    checkpoint_val = ModelCheckpoint(train_generator.save_folder_val,
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     save_best_only=True,
                                     save_freq='epoch',
                                     verbose=1,
                                     mode='min')

    early_stopping = EarlyStopping(monitor='loss',
                                   min_delta=0,
                                   patience=20,
                                   verbose=1,
                                   restore_best_weights=True)


    def lr_schedule(epoch):

        if epoch < 20:
            return 1E-3
        elif epoch < 100:
            return 1E-3
        elif epoch < 200:
            return 1E-3
        else:
            return 1E-4


    learning_schedule = CustomLearningRateScheduler(lr_schedule)

    csv_logger = CSVLogger('log.csv', append=False)

    call_backs = [checkpoint,
                  checkpoint_val,
                  csv_logger,
                  learning_schedule]

    loss = LossYolo3(train_generator.input_size,
                     train_generator.batch_size,
                     train_generator.anchors_array,
                     train_generator.pattern_shape,
                     train_generator.classes,
                     iou_ignore_thresh=0.5,
                     coord_scale=train_generator.lambda_coord,
                     class_scale=train_generator.lambda_class,
                     obj_scale=train_generator.lambda_object,
                     noobj_scale=train_generator.lambda_no_object)

    model.compile(optimizer=Nadam(learning_rate=train_generator.learning_rate),
                  loss=loss)

    model.fit(train_generator.yield_next_batch(),
              steps_per_epoch=train_generator.steps_per_epoch,
              validation_data=valid_generator.yield_next_batch(),
              validation_steps=valid_generator.steps_per_epoch,
              epochs=train_generator.epoch,
              callbacks=call_backs,
              workers=1,
              shuffle=True,
              max_queue_size=8,
              initial_epoch=0)
