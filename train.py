from tensorflow.python.keras.callbacks import *
from tensorflow.keras.optimizers import *

from Config.train_config import *
from DataSet.batch_generator import BatchGenerator
from Loss.loss import Loss
from Nets.yolo3_net import get_yolo3_backend

if __name__ == "__main__":
    config_file = r"config\pascal_voc.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])

    # Create Data
    train_generator = BatchGenerator(model_cfg, train_cfg, True)
    val_generator = BatchGenerator(model_cfg, train_cfg, False)

    # Create Mode
    model = get_yolo3_backend((model_cfg.input_size, model_cfg.input_size), model_cfg.classes)

    checkpoint = ModelCheckpoint(train_cfg.pretrain_weight,
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

    csv_logger = CSVLogger('shuffle256.csv', append=False)

    current_net_size = (train_generator.input_size, train_generator.input_size)

    loss_bin = Loss(current_net_size,
                    train_generator.batch_size,
                    train_generator.anchors_array,
                    train_generator.pattern_shape)

    model.compile(optimizer=RMSprop(lr=1E-3),
                  loss=loss_bin)

    model.fit(train_generator.get_next_batch(),
              steps_per_epoch=train_generator.steps_per_epoch,
              validation_data=val_generator.get_next_batch(),
              validation_steps=val_generator.steps_per_epoch,
              epochs=100,
              callbacks=[checkpoint, reduce_lr, early_stopping, csv_logger],
              initial_epoch=0)
