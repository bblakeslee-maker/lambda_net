# LambdaNet: A Fully Convolutional Architecture for Directional Change Detection
This repository contains the official codebase for LambdaNet.

## Requirements
LambdaNet was developed in Python 3.6 and relies on the dependencies listed in the requriements.txt file.  A GPU is required for both training and inference.

## Installation
First, clone the repository using the following command:

```git clone https://bblakeslee411@bitbucket.org/bblakeslee411/lambdanet.git```

Next, install the dependencies with:

```pip3 install -r requirements.txt```

## Usage
LambdaNet is capable of operating in both binary and multi-class change detection modes.  Configuration of the model is handled through the config files located in the ```cfgs``` folder.  Sample config files and documentation can be found inside the ```cfgs``` folder.

For the binary case, the training and evaluation invocations are:

```python3 train.py --config <configName> --modelName <modelName>```

```python3 eval.py  --config <configName> --modelName <modelName> --resultName <resultName>```

For the multi-class case, the training and evaluation invocations are:

```python3 multiClassTrain.py --config <configName> --modelName <modelName>```

```python3 multiClassEval.py  --config <configName> --modelName <modelName> --resultName <resultName>```

### Flags
The following flags are used to invoke LambdaNet:

* ```--config```:  (Required) Specifies the config file to use in training the network.

* ```--modelName```:  (Required) Specified the name of the model.  For training, this will be the model name of the final model when saved.  For inference, this will be the name of the model to load.

* ```--resultName```:  (Required) Inference only.  Specifies the name of the results folder to store imagery and statistical data in.

* ```--gpu <gpuId>```: (Optional) Specifies the GPU to place the model on.  Omitting this flag will place the model on the first available GPU.

### Results
Models, configuration information, and results are stored under the ```models``` folder.  The subsequent directory structure is generated automatically as models are trained and inferenced.  A sample directory structure is shown below (Comments in parentheses):

```
models
	|- modelNameA (From --modelName flag.)
		|- modelNameA.pth (Saved model)
		|- configFile.txt (Copy of configuration file used to train model.)
		|- modelNameA_loss.log (Loss record of training process. Formatted as JSON.)
		|- results
			|- resultNameA (From --resultName flag.)
				|- imageFiles (Result images.)
				|- globalStats.txt (Global statistics for change masks.)
```

Results imagery is formatted as follows:

*  Upper Left: Past input image

*  Upper Right: Present input image

*  Lower Left: Ground truth

*  Lower Right: Predicted changes

#### Structured Change Detection Results
These images illustrate LambdaNet operating in the purely structured mode.

![Structured Directional Change Results 1](/demoImgs/18312.png)
![Structured Directional Change Results 2](/demoImgs/18312.png)

#### Unstructured Change Detection Results
These images illustrate LambdaNet operating in mixed mode.  People are marked as structured changes, while the unlabelled bottles are marked as unstructured changes.
![Unstructured Directional Change Results 1](/demoImgs/17890.png)
![Unstructured Directional Change Results 2](/demoImgs/17891.png)

## Configuration File Documentation
This file contains a listing of all config file headers and flags used in training and inferencing LambdaNet.

### [paths] Subsection
This subsection contains configuration information for the file paths used to train and evaluate LambdaNet.

* ```datapath```: Path to top level folder of dataset.

* ```trainsplit```: Path to file containing the list of files in the training split.

* ```valsplit```: Path to file containing the list of files in the validation split.

* ```testsplit```: Path to file containing the list of files in the test split.

### [arch] Subsection
This subsection details the architectural constraints for LambdaNet.

* ```outputchannels```: Number of output channels for the decoder network.  A single output channel indicates binary classification mode, while four output channels are used for directional change mode.

* ```encodermodel```: Name of the encoder model to use.

* ```decodermodel```: Name of the decoder model to use.

* ```fusiontype```: Type of fusion node used to unify the output maps from the encoders.  Currently supports ```cat``` (concatenation) and ```diff``` (difference) modes.

* ```mode```: Default to ```lambda```.

* ```normtype```: Type of normalization to use.  Use ```batch``` for batch normalization and ```inst``` for instance normalization.

### [params] Subsection
This subsection lists the hyperparameters used to tune LambdaNet.

* ```numepochs```: Number of epochs to train model for.

* ```randominputswap```: Indicates whether or not to randomly swap the input images to avoid overfitting.  Either True or False.

* ```learningrate```: Learning rate for model training.

* ```batchsize```: Image batch size.

* ```decayinterval```: Number of epochs to wait between running the learning rate scheduler to scale the specified learning rate.

* ```decayrate```: Scale factor used to scale the learning rate.

* ```addchangeweight```: Gradient scale factor associated with the additive change class.

* ```subchangeweight```: Gradient scale factor associated with the subtractive change class.

* ```exchangeweight```: Gradient scale factor associated with the exchange class.

### [preprocess] Subsection
This subsection contains the image preprocessing parameters for LambdaNet.

* ```imgheight```: Height to resize input images to.

* ```imgwidth```: Width to resize input images to.

* ```redmean```: Average value of the input image's red channel.

* ```greenmean```: Average value of the input image's green channel.

* ```bluemean```: Average value of the input image's blue channel.

## Split File Documentation
This file contains a listing of all image pairs contained in a given split.  Each line should be formatted as follows:

```relative/path/to/past/image,relative/path/to/present/image,relative/path/to/ground/truth```

Each path should be relative to the top level directory specified in the datapath entry of the paths subsection of the configuration file.  When an entry in the split file is concatenated with the datapath top level directory, it should form an absolute path to the file.

## Extension of Code
The LambdaNet codebase is written to be modular.  To develop new architectures, it should only be necessary to create new encoder or decoder files.  For interface information, see the sample files provided in the ```encoders``` and ```decoders``` folders.  New fusion nodes may be implemented by modifying the ```lambdaShell``` file, located under the ```utility``` folder.