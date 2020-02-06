import os
import warnings

from utils.eval_utils import save_confusion_matrix, predictive_entropy, mutual_info, \
    uncertainty_fraction_removal, normalized_uncertainty_toleration_removal
from utils.py_utils import load_data, softmax
import numpy as np
import h5py
from utils.eval_utils import class_names
warnings.filterwarnings("ignore")
from config import options
from models import backbone

if __name__ == '__main__':
    ##################################
    # Initialize the directories
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(BASE_DIR, options.save_dir, options.model_name, 'inference_results')
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    ##################################
    # Load the data
    ##################################
    data_dir = os.path.join(BASE_DIR, 'preprocessed_data.h5')
    x_train, y_train, x_valid, y_valid = load_data(data_dir)

    if not options.MC:  # normal evaluation
        model = backbone(options.backbone_name).classification_model(load_weights_from=options.model_name,
                                                                     num_classes=options.num_classes,
                                                                     num_dense_layers=options.num_dense_layers,
                                                                     num_dense_units=options.num_dense_units,
                                                                     dropout_rate=0.,
                                                                     pooling=options.pooling,
                                                                     kernel_regularizer=options.dense_layer_regularizer)
        y_pred = model.predict(x_valid)
        y_prob = softmax(y_pred)

        fig_path = os.path.join(dest_path, 'confusion_matrix' + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), y_prob.argmax(axis=1).astype(int),
                              classes=np.array(class_names),
                              dest_path=fig_path,
                              title='Confusion matrix, without normalization')

        fig_path = os.path.join(dest_path, 'normalized_confusion_matrix' + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), y_prob.argmax(axis=1).astype(int),
                              classes=np.array(class_names), normalize=True,
                              dest_path=fig_path,
                              title='Normalized confusion matrix')

        acc = 1 - np.count_nonzero(np.not_equal(y_prob.argmax(axis=1), y_valid.argmax(axis=1))) / y_prob.shape[0]
        print('accuracy={0:.02%}'.format(acc))

        file_name = os.path.join(dest_path, 'predictions.h5')
        h5f = h5py.File(file_name, 'w')
        h5f.create_dataset('x_valid', data=x_valid)
        h5f.create_dataset('y_valid', data=y_valid)
        h5f.create_dataset('y_prob', data=y_prob)
        h5f.close()

    else:  # MC-evaluation
        model = backbone(options.backbone_name).classification_model(load_weights_from=options.model_name,
                                                                     num_classes=options.num_classes,
                                                                     num_dense_layers=options.num_dense_layers,
                                                                     num_dense_units=options.num_dense_units,
                                                                     pooling=options.pooling,
                                                                     dropout_rate=options.dropout_rate,
                                                                     kernel_regularizer=options.dense_layer_regularizer,
                                                                     print_model_summary=True)

        y_prob_mc = np.zeros((options.mc_simulations, y_valid.shape[0], options.num_classes))
        for mc_iter in range(options.mc_simulations):
            print('running iter#{}'.format(mc_iter))
            y_prob_mc[mc_iter] = model.predict(x_valid)

        mean_prob = y_prob_mc.mean(axis=0)  # prediction probabilities of shape [2005, 7]
        y_pred = mean_prob.argmax(axis=1)  # predicted class labels of shape [2005,]
        var_pred_entropy = predictive_entropy(mean_prob)
        var_pred_MI = mutual_info(y_prob_mc)
        acc = 1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), y_valid.argmax(axis=1))) / mean_prob.shape[0]
        print('accuracy={0:.02%}'.format(acc))

        fig_path = os.path.join(dest_path, 'confusion_matrix_MC=' + str(options.mc_simulations) + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), mean_prob.argmax(axis=1).astype(int),
                              classes=np.array(class_names),
                              dest_path=fig_path,
                              title='Confusion matrix, without normalization')

        fig_path = os.path.join(dest_path, 'normalized_confusion_matrix_MC=' + str(options.mc_simulations) + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), mean_prob.argmax(axis=1).astype(int),
                              classes=np.array(class_names), normalize=True,
                              dest_path=fig_path,
                              title='Normalized confusion matrix')

        # plot and save the uncertainty toleration removal figure
        num_intervals = 20
        H = var_pred_entropy
        H_norm = (H - H.min()) / (H.max() - H.min())    # normalize the uncertainty values (in range [0, 1])
        normalized_uncertainty_toleration_removal(y_valid.argmax(axis=1), y_pred, H_norm, num_intervals,
                                                  save=True, save_dir=dest_path)

        reps_for_random = 40
        num_fractions = 20
        uncertainty_fraction_removal(y_valid.argmax(axis=1), y_pred, H_norm, num_fractions, reps_for_random,
                                     save=True, save_dir=dest_path)

        file_name = os.path.join(dest_path, 'MC_predictions.h5')
        h5f = h5py.File(file_name, 'w')
        h5f.create_dataset('x_valid', data=x_valid)
        h5f.create_dataset('y_valid', data=y_valid)
        h5f.create_dataset('y_prob_mc', data=y_prob_mc)
        h5f.create_dataset('mean_prob', data=mean_prob)
        h5f.create_dataset('var_pred_entropy', data=var_pred_entropy)
        h5f.create_dataset('var_pred_MI', data=var_pred_MI)
        h5f.close()
