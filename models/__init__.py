class Backbone(object):
    """
    This class stores additional information on backbones
    """

    def __init__(self, backbone_name, **kwargs):
        """
        :param backbone_name: name of the backbone
        :param kwargs: user provided kwargs in case a custom base model is needed
        """
        from utils.metrics_utils import dice_coeff
        from utils.metrics_utils import jaccard_index
        from utils.metrics_utils import pixelwise_precision
        from utils.metrics_utils import pixelwise_specificity
        from utils.metrics_utils import pixelwise_sensitivity
        from utils.initializers import PriorProbability

        self.custom_objects = {
            # included custom metrics in case the saved model need to be compiled
            'dice_coeff': dice_coeff,
            'jaccard_index': jaccard_index,
            'pixelwise_precision': pixelwise_precision,
            'pixelwise_specificity': pixelwise_specificity,
            'pixelwise_sensitivity': pixelwise_sensitivity,
            'PriorProbability': PriorProbability,
        }

        self.backbone_name = backbone_name
        self.backbone_options = kwargs
        self.scale_factor = 2
        self.validate()

    def build_base_model(self, inputs, **kwarg):
        raise NotImplementedError('backbone method not implemented')

    def classification_model(self,
                             input_shape=None,
                             input_padding=None,
                             submodel=None,
                             num_classes=7,
                             num_dense_layers=2,
                             num_dense_units=256,
                             dropout_rate=0.,
                             pooling=None,
                             use_output_activation=True,
                             kernel_regularizer=None,
                             use_activation=True,
                             include_top=True,
                             name='default_classification_model',
                             print_model_summary=False,
                             plot_model_summary=False,
                             load_from=None,
                             load_model_from=None,
                             load_weights_from=None,
                             save_to=None,
                             lr=1e-5,
                             ):
        """ Returns a classifier model using the correct backbone.
        """
        import keras
        from keras import backend as K
        from models.submodels.classification import default_classification_model
        from utils.model_utils import plot_model
        from utils.model_utils import load_model_from_run
        from utils.model_utils import save_model_to_run
        from utils.model_utils import load_model_weights_from
        from utils.print_utils import on_aws

        if load_from:
            model = load_model_from_run(self.backbone_name, load_from, load_from)
        elif load_model_from:
            model = load_model_from_run(self.backbone_name, load_model_from, load_weights_from)
        else:
            if K.image_data_format() == 'channels_last':
                input_shape = (224, 224, 3) if input_shape is None else input_shape
            else:
                input_shape = (3, 224, 224) if input_shape is None else input_shape

            inputs = keras.layers.Input(shape=input_shape)

            x = inputs
            if input_padding is not None:
                x = keras.layers.ZeroPadding2D(padding=input_padding)(x)

            base_model = self.build_base_model(inputs=x, **self.backbone_options)
            x = base_model.output

            if submodel is None:
                outputs = default_classification_model(input_tensor=x,
                                                       input_shape=base_model.output_shape[1:],
                                                       num_classes=num_classes,
                                                       num_dense_layers=num_dense_layers,
                                                       num_dense_units=num_dense_units,
                                                       dropout_rate=dropout_rate,
                                                       pooling=pooling,
                                                       use_output_activation=use_activation,
                                                       kernel_regularizer=kernel_regularizer)
            else:
                outputs = submodel(x)

            model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)

            if load_weights_from:
                load_model_weights_from(model, load_weights_from, skip_mismatch=False)

        if print_model_summary:
            model.summary()

        if plot_model_summary and not on_aws():
            plot_model(save_to_dir=save_to, model=model, name=name)

        if save_to:
            save_model_to_run(model, save_to)

        compile_model(model=model,
                      num_classes=num_classes,
                      metrics='acc',
                      loss='ce',
                      lr=lr)

        return model

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('backbone method not implemented')


def backbone(backbone_name, **kwargs):
    """
    Returns a backbone object for the given backbone.
    """
    if 'vgg' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'unet' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'inception' in backbone_name:
        from .inception import InceptionBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    elif 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))

    return b(backbone_name, **kwargs)


def compile_model(model, num_classes, metrics, loss, lr):
    from keras.losses import binary_crossentropy
    from keras.losses import categorical_crossentropy

    from keras.metrics import binary_accuracy
    from keras.metrics import categorical_accuracy

    from keras.optimizers import Adam

    from utils.metrics_utils import dice_coeff
    from utils.metrics_utils import jaccard_index
    from utils.metrics_utils import class_jaccard_index
    from utils.metrics_utils import pixelwise_precision
    from utils.metrics_utils import pixelwise_sensitivity
    from utils.metrics_utils import pixelwise_specificity
    from utils.metrics_utils import pixelwise_recall

    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            if num_classes == 1:
                loss = binary_crossentropy
            else:
                loss = categorical_crossentropy
        else:
            raise ValueError('unknown loss %s' % loss)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metrics[i] = binary_accuracy if num_classes == 1 else categorical_accuracy
        elif metric == 'jaccard_index':
            metrics[i] = jaccard_index(num_classes)
        elif metric == 'jaccard_index0':
            metrics[i] = class_jaccard_index(0)
        elif metric == 'jaccard_index1':
            metrics[i] = class_jaccard_index(1)
        elif metric == 'jaccard_index2':
            metrics[i] = class_jaccard_index(2)
        elif metric == 'jaccard_index3':
            metrics[i] = class_jaccard_index(3)
        elif metric == 'jaccard_index4':
            metrics[i] = class_jaccard_index(4)
        elif metric == 'jaccard_index5':
            metrics[i] = class_jaccard_index(5)
        elif metric == 'dice_coeff':
            metrics[i] = dice_coeff(num_classes)
        elif metric == 'pixelwise_precision':
            metrics[i] = pixelwise_precision(num_classes)
        elif metric == 'pixelwise_sensitivity':
            metrics[i] = pixelwise_sensitivity(num_classes)
        elif metric == 'pixelwise_specificity':
            metrics[i] = pixelwise_specificity(num_classes)
        elif metric == 'pixelwise_recall':
            metrics[i] = pixelwise_recall(num_classes)
        else:
            raise ValueError('metric %s not recognized' % metric)

    model.compile(optimizer=Adam(lr=lr),
                  loss=loss,
                  metrics=metrics)

