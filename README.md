# Handwriting_Recognition_CRNN_LSTM
In this notebook, we'll go through the steps to train a CRNN (CNN+RNN) model for handwriting recognition. The model will be trained using the CTC(Connectionist Temporal Classification) loss function
# Why Deep Learning?
![WhyDeepLearning](https://user-images.githubusercontent.com/67474853/136068397-89e3aedf-09b0-4d5b-b2ee-91bc00881377.png)
Deep Learning self extracts features with a deep neural networks and classify itself. Compare to traditional Algorithms it performance increase with Amount of Data.
## Methodology
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
![Screenshot from 2021-10-06 01-51-58](https://user-images.githubusercontent.com/67474853/136097194-054a2f14-58fd-4bd8-ab62-b1c1081aeb84.png)
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
}
```
### Training Model ###
![Screenshot from 2021-10-06 02-03-07](https://user-images.githubusercontent.com/67474853/136098912-b66196de-6d63-4cc4-9f62-92de64f63fd9.png)
### Decoding Outputs from Prediction ###
*Performance Check:(Levenshtein Distance)
 For computing the performance, using the Jaro-Winkler algorithm to detect similarity between the captured text and the actual text.
![Screenshot from 2021-10-06 01-59-10](https://user-images.githubusercontent.com/67474853/136098042-dfd93d3d-44c4-45dd-af96-26b77ddd4bf7.png)
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
