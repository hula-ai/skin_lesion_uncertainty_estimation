import os
import warnings
import sys
warnings.filterwarnings("ignore")
from datetime import datetime
from config import options
from models import backbone
from keras.preprocessing.image import ImageDataGenerator
from utils.callback_utils import config_cls_callbacks
from utils.eval_utils import compute_class_weights
from utils.filename_utils import get_log_filename
from utils.print_utils import log_variable, Tee
from utils.py_utils import load_data

if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + options.backbone_name
    logfile = open(get_log_filename(run_name=save_dir), 'w+')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

    ##################################
    # Load the data
    ##################################
    data_dir = os.path.join(BASE_DIR, 'preprocessed_data.h5')
    x_train, y_train, x_valid, y_valid = load_data(data_dir)

    ##################################
    # Create the model
    ##################################
    callbacks = config_cls_callbacks(save_dir)
    backbone_options = {}

    model = backbone(options.backbone_name, **backbone_options).classification_model(
        input_shape=x_train.shape[1:],
        num_classes=options.num_classes,
        num_dense_layers=options.num_dense_layers,
        num_dense_units=options.num_dense_units,
        pooling=options.pooling,
        dropout_rate=options.dropout_rate,
        kernel_regularizer=options.dense_layer_regularizer,
        save_to=save_dir,
        load_from=None,
        print_model_summary=True,
        plot_model_summary=False,
        lr=options.lr)

    class_weights = compute_class_weights(y_train, wt_type=options.class_wt_type)

    log_variable(var_name='num_dense_layers', var_value=options.num_dense_layers)
    log_variable(var_name='num_dense_units', var_value=options.num_dense_units)
    log_variable(var_name='dropout_rate', var_value=options.dropout_rate)
    log_variable(var_name='pooling', var_value=options.pooling)
    log_variable(var_name='class_wt_type', var_value=options.class_wt_type)
    log_variable(var_name='dense_layer_regularizer', var_value=options.dense_layer_regularizer)
    log_variable(var_name='class_wt_type', var_value=options.class_wt_type)
    log_variable(var_name='learning_rate', var_value=options.lr)
    log_variable(var_name='batch_size', var_value=options.batch_size)
    log_variable(var_name='use_data_aug', var_value=options.use_data_aug)

    if options.use_data_aug:
        log_variable(var_name='horizontal_flip', var_value=options.horizontal_flip)
        log_variable(var_name='vertical_flip', var_value=options.vertical_flip)
        log_variable(var_name='width_shift_range', var_value=options.width_shift_range)
        log_variable(var_name='height_shift_range', var_value=options.height_shift_range)
        log_variable(var_name='rotation_angle', var_value=options.rotation_angle)

    log_variable(var_name='n_samples_train', var_value=x_train.shape[0])
    log_variable(var_name='n_samples_valid', var_value=x_valid.shape[0])

    sys.stdout.flush()  # need to make sure everything gets written to file

    if options.use_data_aug:
        datagen = ImageDataGenerator(rotation_range=options.rotation_angle,
                                     horizontal_flip=options.horizontal_flip,
                                     vertical_flip=options.vertical_flip,
                                     width_shift_range=options.width_shift_range,
                                     height_shift_range=options.height_shift_range)

        model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=options.batch_size),
                            steps_per_epoch=x_train.shape[0] // options.batch_size,
                            epochs=options.epochs,
                            initial_epoch=0,
                            verbose=1,
                            validation_data=(x_valid, y_valid),
                            callbacks=callbacks,
                            workers=8,
                            use_multiprocessing=True)

    else:

        model.fit(x=x_train,
                  y=y_train,
                  batch_size=options.batch_size,
                  epochs=options.epochs,
                  verbose=1,
                  validation_data=(x_valid, y_valid),
                  class_weight=class_weights,
                  shuffle=True,
                  callbacks=callbacks)

    sys.stdout = original

