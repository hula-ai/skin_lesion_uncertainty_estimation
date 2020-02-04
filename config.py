from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=500, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=1, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=2, type='int',
                  help='frequency of displaying the training results (default: 100)')

# DATA
parser.add_option('--ih', '--img_h', dest='img_h', default=512, type='int',
                  help='input image height (default: 28)')
parser.add_option('--iw', '--img_w', dest='img_w', default=512, type='int',
                  help='input image width (default: 28)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 1)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=7, type='int',
                  help='number of classes (default: 10)')

# DATA AUGMENTATION
parser.add_option('--uda', '--use_data_aug', dest='use_data_aug', default=True,
                  help='whether to use data augmentation or not (default: True)')
parser.add_option('--hf', '--horizontal_flip', dest='horizontal_flip', default=True,
                  help='whether to use horizontal flipping or not (default: True)')
parser.add_option('--vf', '--vertical_flip', dest='vertical_flip', default=True,
                  help='whether to use vertical flipping or not (default: True)')
parser.add_option('--ra', '--rotation_angle', dest='rotation_angle', default=180, type='int',
                  help='rotation angle in degrees (default: 180)')
parser.add_option('--wsr', '--width_shift_range', dest='width_shift_range', default=0.1, type='float',
                  help='width shift range (default: 0.1)')
parser.add_option('--hsr', '--height_shift_range', dest='height_shift_range', default=0.1, type='float',
                  help='height shift range (default: 0.1)')

# MODEL
parser.add_option('--bbn', '--backbone_name', dest='backbone_name', default='densenet169',
                  help='vgg16, resnet50, densenet169, densenet201, inception_v3 (default: densenet169)')
parser.add_option('--poo', '--pooling', dest='pooling', default='avg',
                  help='Pooling layer at the end of CNN; avg or max (default: avg)')
parser.add_option('--ndl', '--num_dense_layers', dest='num_dense_layers', default=1, type='int',
                  help='number of Dense layers at the end of network (default: 1)')
parser.add_option('--ndu', '--num_dense_units', dest='num_dense_units', default=128, type='int',
                  help='number of Dense layers at the end of network (default: 1)')
parser.add_option('--dr', '--dropout_rate', dest='dropout_rate', default=0.5, type='float',
                  help='learning rate (default: 0.001)')
parser.add_option('--dlr', '--dense_layer_regularizer', dest='dense_layer_regularizer', default='L2',
                  help='None, L1 or L2 (default: L2)')

# LOSS and OPTIMIZER
parser.add_option('--cwt', '--class_wt_type', dest='class_wt_type', default='balanced',
                  help='ones, balanced, balanced_sqrt (default: balanced)')
parser.add_option('--lr', '--lr', dest='lr', default=1e-4, type='float',
                  help='learning rate (default: 0.001)')


# SAVE and LOAD
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/skin_lesion_uncertainty_estimation/save/'
                          '20200123_183951/models/18122.ckpt',
                  help='path to load a .ckpt model')

options, _ = parser.parse_args()
