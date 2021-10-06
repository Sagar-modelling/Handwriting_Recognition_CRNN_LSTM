# Handwriting_Recognition using CRNN_CTC architecture for an deep-learning-based OCR Model.
# Introduction
OCR = Optical Character Recognition. In other words, OCR systems transform a two-dimensional image of text, that could contain machine printed or handwritten text from its image representation into machine-readable text. OCR as a process generally consists of several sub-processes to perform as accurately as possible. The subprocesses are:
* Preprocessing of the Image
* Line Segmentation
* Word Segmentation
* Text Recognition
* Post Processing
# Problem Statement
For almost two decades, optical character recognition systems have been widely used to provide automated text entry into computerized systems. Yet in all this time, conventional OCR systems (like zonal OCR) have never overcome their inability to read more than a handful of type fonts and page formats.

Next-generation OCR engines deal with these problems mentioned above really good by utilizing the latest research in the area of deep learning. By leveraging the combination of deep models and huge datasets publicly available, models achieve state-of-the-art accuracies on given tasks. Nowadays it is also possible to generate synthetic data with different fonts using generative adversarial networks and few other generative approaches.

Optical Character Recognition remains a challenging problem when text occurs in unconstrained environments, like natural scenes, due to geometrical distortions, complex backgrounds, and diverse fonts. The technology still holds an immense potential due to the various use-cases of deep learning based OCR like
* building license plate readers
* digitizing invoices
* digitizing menus
* digitizing ID cards
* 
In this notebook, we'll go through the steps to train a CRNN (CNN+RNN) model for handwriting recognition. The model will be trained using the CTC(Connectionist Temporal Classification) loss function.
# Why Deep Learning?
![WhyDeepLearning](https://user-images.githubusercontent.com/67474853/136068397-89e3aedf-09b0-4d5b-b2ee-91bc00881377.png)
Deep Learning self extracts features with a deep neural networks and classify itself. Compare to traditional Algorithms it performance increase with Amount of Data.
## Methodology for text recognition
The handwriting recognition model which takes a line as input and converts the line into digital text. This model consits of a CNN-biLSTM architecture. The loss used is the CTC (Connectionist Temporal Classification) loss.

![1_Uof8K-rRMKJTCtAtO1LypQ](https://user-images.githubusercontent.com/67474853/136075952-26dadc77-4ac2-4c9a-b339-c2509d4eed85.png)

Here is the CNN-biLSTM architecture model(Detailed Project Workflow).

The input lines are sent into the CNN to extract features from similar patterns. These image features are then sent to a sequential learner which are the bidirectional LSTMs which are then sent to the output string that predict the character based on the alphabet with the highest predicted value given by the model.

Project consists of Three steps:
* Multi-scale feature Extraction --> Convolutional Neural Network 7 Layers.
* Sequence Labeling (BLSTM-CTC) --> Recurrent Neural Network (2 layers of LSTM) with CTC.
* Transcription --> Decoding the output of the RNN (CTC decode).

We can break the implementation of CRNN network into following steps:

### Import Dependencies and Data ###
* Import data using Pandas Dataframe.
* Only needed the words images and words.txt.
* Place the downloaded files inside data directory.

### Data Preprocessing and preparing the images for training ###
* The images are loaded as grayscale and reshaped to width 128 and height 32.
* The width and height are cropped if they are greater than 128 and 32 respectively. If they are smaller, then the image is padded with white pixels. Finally the   image is rotated clockwise to bring the image shape to (x, y).
* The image is then normalized to range [0, 1]

### Label Encoding for CTC Loss ###
* Learn more about CTC loss and why its amazing for text recognition from [here](https://towardsdatascience.com/handwriting-to-text-conversion-using-time-distributed-cnn-and-lstm-with-ctc-loss-function-a784dccc8ec3).
* The labels have to be converted to numbers which represent each character in the training set. The 'alphabets' consist of A-Z and three special characters (- ' and space).

### Model Building ###
* Input shape for our architecture having an input image of height 32 and width 128.
* Here we used seven convolution layers of which 6 are having kernel size (3,3) and the last one is of size (2.2) and the number of filters is increased from 64    to 512 layer by layer.
* Two max-pooling layers are added with size (2,2) and then two max-pooling layers of size (2,1) are added to extract features with a larger width to predict long   texts.
* Also, we used batch normalization layers after fifth and sixth convolution layers which accelerates the training process.
* Then we used a lambda function to squeeze the output from conv layer and make it compatible with LSTM layer.
* Then used two Bidirectional LSTM layers each of which has 128 units. This RNN layer gives the output of size (batch_size, 31, 79). Where 79 is the total number   of output classes including blank character.
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, 128, 1)]      0         
_________________________________________________________________
conv2d (Conv2D)              (None, 32, 128, 64)       640       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 64, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 64, 128)       73856     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 32, 128)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 32, 256)        295168    
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 32, 256)        590080    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 32, 256)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 32, 512)        1180160   
_________________________________________________________________
batch_normalization (BatchNo (None, 4, 32, 512)        2048      
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 32, 512)        2359808   
_________________________________________________________________
batch_normalization_1 (Batch (None, 4, 32, 512)        2048      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 32, 512)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 31, 512)        1049088   
_________________________________________________________________
lambda (Lambda)              (None, 31, 512)           0         
_________________________________________________________________
bidirectional (Bidirectional (None, 31, 512)           1574912   
_________________________________________________________________
bidirectional_1 (Bidirection (None, 31, 512)           1574912   
_________________________________________________________________
dense (Dense)                (None, 31, 79)            40527     
=================================================================
Total params: 8,743,247
Trainable params: 8,741,199
Non-trainable params: 2,048
```
### Defining Loss Function ###
```
the_labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])

#model to be used at training time
model = Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)
```
### Training Model ###
```
batch_size = 5
epochs = 25
e = str(epochs)
optimizer_name = 'sgd'

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = optimizer_name, metrics=['accuracy'])

filepath="{}o-{}r-{}e-{}t-{}v.hdf5".format(optimizer_name,
                                          str(RECORDS_COUNT),
                                          str(epochs),
                                          str(train_images.shape[0]),
                                          str(valid_images.shape[0]))

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

history = model.fit(x=[train_images, train_padded_label, train_input_length, train_label_length],
                    y=np.zeros(len(train_images)),
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=([valid_images, valid_padded_label, valid_input_length, valid_label_length], [np.zeros(len(valid_images))]),
                    verbose=1,
                    callbacks=callbacks_list)
```
### Decoding Outputs from Prediction ###
*Performance Check:(Levenshtein Distance)
 For computing the performance, using the Jaro-Winkler algorithm to detect similarity between the captured text and the actual text.
```
filepath='./sgdo-30000r-25e-18074t-2007v.hdf5'
# load the saved best model weights
act_model.load_weights(filepath)

# predict outputs on validation images
prediction = act_model.predict(valid_images)
 
# use CTC decoder
decoded = K.ctc_decode(prediction, 
                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                       greedy=True)[0][0]
out = K.get_value(decoded)

import Levenshtein as lv

total_jaro = 0

# see the results
for i, x in enumerate(out):
    letters=''
    for p in x:
        if int(p) != -1:
            letters+=char_list[int(p)]
    total_jaro+=lv.jaro(letters, valid_original_text[i])
  
print('jaro :', total_jaro/len(out))
```
```output
jaro : 0.9361624272311879
```
# Requirements(Dependencies)
* Tensorflow 1.8.0
* Numpy
* OpenCv
* Pandas
* matplotlib
* sklearn
## Dataset Used ##
* IAM dataset download from [here](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).
# Prediction output on IAM Test Data
![Screenshot from 2021-10-06 02-13-54](https://user-images.githubusercontent.com/67474853/136099914-e5458ced-1915-49a8-8716-d649bcfe2dcf.png)
# Prediction output on Self Test Data
![Screenshot from 2021-10-06 02-28-21](https://user-images.githubusercontent.com/67474853/136101892-baf13b9d-8d19-413d-85fe-6200bd9ada0c.png)


![Screenshot from 2021-10-06 02-28-41](https://user-images.githubusercontent.com/67474853/136101933-b1ce90a0-f7b2-46d6-9945-759677061079.png)
# Further Improvement
* Line segementation can be added for full paragraph text recognition. For line segmentation you can use A* path planning algorithm or CNN model or opencv to  
  seperate paragraph into lines.    
* Better Image preprocessing such as: reduce backgoround noise to handle real time image more accurately.
* Better Decoding approach to improve accuracy. Some of the CTC Decoder found [here](https://github.com/githubharald/CTCDecoder).
* Using MxNet deep learning framework and [MDLSTM](https://arxiv.org/abs/1604.03286) to recognize whole paragraph at once Scan, Attend and Read: End-to-End  Handwritten Paragraph Recognition.
* Modifying and extending the CRNN+LSTM+CTC architecture for Hindi Handwriting text segmentation and recognition.
