import os
#if os.system('nvidia-smi') == 0:
#    import setGPU
import tensorflow as tf
import glob
import sys
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import resnet_v1_eembc
from data_loader import nEXODataset
import yaml
import csv
# from keras_flops import get_flops # (different flop calculation)
import kerop
from tensorflow.data import Dataset

def get_lr_schedule_func(initial_lr, lr_decay):

    def lr_schedule_func(epoch):
        return initial_lr * (lr_decay ** epoch)

    return lr_schedule_func


def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param


def main(args):

    # parameters
    config = yaml_load(args.config)
    data_name = config['data']['name']
    h5file = config['data']['h5name']
    csv_train = config['data']['train_csv']
    csv_test = config['data']['test_csv']
    input_shape = [int(i) for i in config['data']['input_shape']]
    num_classes = int(config['data']['num_classes'])
    num_filters = config['model']['filters']
    kernel_sizes = config['model']['kernels']
    strides = config['model']['strides']
    l1p = float(config['model']['l1'])
    l2p = float(config['model']['l2'])
    skip = bool(config['model']['skip'])
    avg_pooling = bool(config['model']['avg_pooling'])
    batch_size = config['fit']['batch_size']
    num_epochs = config['fit']['epochs']
    verbose = config['fit']['verbose']
    patience = config['fit']['patience']
    save_dir = config['save_dir']
    model_name = config['model']['name']
    loss = config['fit']['compile']['loss']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    # quantization parameters
    if 'quantized' in model_name:
        logit_total_bits = config["quantization"]["logit_total_bits"]
        logit_int_bits = config["quantization"]["logit_int_bits"]
        activation_total_bits = config["quantization"]["activation_total_bits"]
        activation_int_bits = config["quantization"]["activation_int_bits"]
        alpha = config["quantization"]["alpha"]
        use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
        logit_quantizer = config["quantization"]["logit_quantizer"]
        activation_quantizer = config["quantization"]["activation_quantizer"]
        final_activation = bool(config['model']['final_activation'])

    # optimizer
    optimizer = getattr(tf.keras.optimizers, config['fit']['compile']['optimizer'])
    initial_lr = config['fit']['compile']['initial_lr']
    lr_decay = config['fit']['compile']['lr_decay']

    # load dataset
    #csv_train = '/expanse/lustre/scratch/zli10/temp_project/hls4ml/nexo_train.csv' 
    #csv_test = '/expanse/lustre/scratch/zli10/temp_project/hls4ml/nexo_valid.csv' 
    #h5file = '/expanse/lustre/scratch/zli10/temp_project/hls4ml/nexo.h5'
    train_dg = nEXODataset('train',h5file,csv_train)
    test_dg = nEXODataset('test',h5file,csv_test)
    
    train_ds = Dataset.from_generator(train_dg, output_types = (tf.float32, tf.int64) , output_shapes = (tf.TensorShape(input_shape),tf.TensorShape([])))
    test_ds = Dataset.from_generator(test_dg, output_types = (tf.float32, tf.int64) , output_shapes = (tf.TensorShape(input_shape),tf.TensorShape([])))
    train_ds = train_ds.interleave(lambda x, y: tf.data.Dataset.from_tensors((x,y)), cycle_length=4, block_length=16)
    test_ds = test_ds.interleave(lambda x, y: tf.data.Dataset.from_tensors((x,y)), cycle_length=4, block_length=16)
    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    kwargs = {'input_shape': input_shape,
              'num_classes': num_classes,
              'num_filters': num_filters,
              'kernel_sizes': kernel_sizes,
              'strides': strides,
              'l1p': l1p,
              'l2p': l2p,
              'skip': skip,
              'avg_pooling': avg_pooling}

    # pass quantization params
    if 'quantized' in model_name:
        kwargs["logit_total_bits"] = logit_total_bits
        kwargs["logit_int_bits"] = logit_int_bits
        kwargs["activation_total_bits"] = activation_total_bits
        kwargs["activation_int_bits"] = activation_int_bits
        kwargs["alpha"] = None if alpha == 'None' else alpha
        kwargs["use_stochastic_rounding"] = use_stochastic_rounding
        kwargs["logit_quantizer"] = logit_quantizer
        kwargs["activation_quantizer"] = activation_quantizer
        kwargs["final_activation"] = final_activation

    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)
    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    # analyze FLOPs (see https://github.com/kentaroy47/keras-Opcounter)
    layer_name, layer_flops, inshape, weights = kerop.profile(model)

    # visualize FLOPs results
    total_flop = 0
    for name, flop, shape in zip(layer_name, layer_flops, inshape):
        print("layer:", name, shape, " MFLOPs:", flop/1e6)
        total_flop += flop
    print("Total FLOPs: {} MFLOPs".format(total_flop/1e6))

    tf.keras.utils.plot_model(model,
                              to_file="model.png",
                              show_shapes=True,
                              show_dtype=False,
                              show_layer_names=False,
                              rankdir="TB",
                              expand_nested=False)

    # Alternative FLOPs calculation (see https://github.com/tokusumi/keras-flops), ~same answer
    # total_flop = get_flops(model, batch_size=1)
    # print("FLOPS: {} GLOPs".format(total_flop/1e9))
    
    # compile model with optimizer
    model.compile(optimizer=optimizer(learning_rate=initial_lr),
                  loss=loss,
                  metrics=['accuracy'])

    # callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

    lr_schedule_func = get_lr_schedule_func(initial_lr, lr_decay)

    callbacks = [ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True),
                 EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
                 LearningRateScheduler(lr_schedule_func, verbose=verbose),
                 ]

    # train
    history = model.fit(train_ds, 
                        #steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=num_epochs,
                        validation_data=test_ds, 
                        callbacks=callbacks,
                        verbose=verbose)

    # restore "best" model
    model.load_weights(model_file_path)

    iterator = iter(test_ds)
    X_test, y_test = next(iterator)
    # get predictions
    y_pred = model.predict(X_test)

    # evaluate with test dataset and share same prediction results
    evaluation = model.evaluate(X_test, y_test)

    auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

    print('Model test accuracy = %.3f' % evaluation[1])
    print('Model test weighted average AUC = %.3f' % auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
