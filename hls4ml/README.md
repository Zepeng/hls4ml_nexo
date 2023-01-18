# Sources
* Dataset
    * nEXO charge simulation [DOI 10.1088/1748-0221/14/09/P09020]
* Model Topology
    * [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
    * [https://keras.io/api/applications/resnet/](https://keras.io/api/applications/resnet/)
    
# Training details
``` python
#learning rate schedule
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print('Learning rate = %f'%lrate)
    return lrate

#optimizer
optimizer = tf.keras.optimizers.Adam()

#define data generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    #brightness_range=(0.9, 1.2),
    #contrast_range=(0.9, 1.2)
)
```
    
# Performance (floating point model)
* Accuracy
    * 
* AUC
    *

# Performance (quantized tflite model)
* Accuracy
    * 
* AUC
    *

# Install 
* Install minicconda from here: https://docs.conda.io/en/latest/miniconda.html

* Create the environment:
```
conda-env create -f environment.yml
```

* Activate the environment:
``` 
conda activate tiny-mlperf-env
```


* Train tiny:
```
python train_nexo.py -c tiny2_pynq-z2_nexo.yml 
```

* Convert tiny:
```
python convert.py -c tiny2_pynq-z2_nexo.yml 
```
