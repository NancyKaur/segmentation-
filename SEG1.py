# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:03:23 2021

@author: User
"""

# Import header files
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from sklearn.metrics import classification_report, confusion_matrix,plot_roc_curve,roc_curve,accuracy_score,roc_auc_score,RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from pocovidnet import MODEL_FACTORY
from pocovidnet.utils import Metrics
from skimage import io
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, auc
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Suppress logging
tf.get_logger().setLevel('ERROR')

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data_dir', type=str,  default='..\pocovidnet\data\cross_validation')
ap.add_argument('-m', '--model_dir', type=str, default='models')
ap.add_argument('-f', '--fold', type=int, default='4', help='fold to take as test data')
ap.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
ap.add_argument('-ep', '--epochs', type=int, default=100)
ap.add_argument('-bs', '--batch_size', type=int, default=16)
ap.add_argument('-t', '--trainable_base_layers', type=int, default=1)
ap.add_argument('-iw', '--img_width', type=int, default=224)
ap.add_argument('-ih', '--img_height', type=int, default=224)
ap.add_argument('-id', '--model_id', type=str, default='vgg_base')
ap.add_argument('-ls', '--log_softmax', type=bool, default=False)
ap.add_argument('-n', '--model_name', type=str, default='test')
ap.add_argument('-hs', '--hidden_size', type=int, default=64)


#python3 scripts/train_covid19.py --data_dir ../data/cross_validation/ --fold 0 --epochs 2

args = vars(ap.parse_args())


# Initialize hyperparameters
DATA_DIR = args['data_dir']
MODEL_NAME = args['model_name']
FOLD = args['fold']
MODEL_DIR = os.path.join(args['model_dir'], MODEL_NAME, f'fold_{FOLD}')
LR = args['learning_rate']
EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
MODEL_ID = args['model_id']
TRAINABLE_BASE_LAYERS = args['trainable_base_layers']
IMG_WIDTH, IMG_HEIGHT = args['img_width'], args['img_height']
LOG_SOFTMAX = args['log_softmax']
HIDDEN_SIZE = args['hidden_size']

# Check if model class exists
if MODEL_ID not in MODEL_FACTORY.keys():
    raise ValueError(
        f'Model {MODEL_ID} not implemented. Choose from {MODEL_FACTORY.keys()}'
    )

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print(f'Model parameters: {args}')
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print('Loading images...')

imagePaths = list(paths.list_files(DATA_DIR))
print(DATA_DIR)
print(imagePaths)
data = []
labels = []

print(f'selected fold: {FOLD}')

train_labels, test_labels = [], []
train_data, test_data = [], []
# test_files = []


# loop over folds
for imagePath in imagePaths:
    
    #if not imagePath.endswith('gif'):
     #   continue

    path_parts = imagePath.split(os.path.sep)
    # extract the split
    train_test = path_parts[-3][-1]
    # extract the class label from the filename
    label = path_parts[-2]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    #cap = cv2.VideoCapture(imagePath)
    #ret, image = cap.read()
    image = io.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # update the data and labels lists, respectively
    if train_test == str(FOLD):
        test_labels.append(label)
        test_data.append(image)
        # test_files.append(path_parts[-1])
    else:
        train_labels.append(label)
        train_data.append(image)

# Prepare data for model
print(
    f'\nNumber of training samples: {len(train_labels)} \n'
    f'Number of testing samples: {len(test_labels)}'
)

assert len(set(train_labels)) == len(set(test_labels)), (
    'Something went wrong. Some classes are only in train or test data.'
)  # yapf: disable

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
train_data = np.array(train_data) / 255.0
test_data = np.array(test_data) / 255.0
train_labels_text = np.array(train_labels)
test_labels_text = np.array(test_labels)

num_classes = len(set(train_labels))

# perform one-hot encoding on the labels
lb = LabelBinarizer()
lb.fit(train_labels_text)

train_labels = lb.transform(train_labels_text)
test_labels = lb.transform(test_labels_text)

if num_classes == 2:
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

trainX = train_data
trainY = train_labels
testX = test_data
testY = test_labels
print('Class mappings are:', lb.classes_)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=10,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

#####################################################################################################
# Load the VGG16 network
model = MODEL_FACTORY[MODEL_ID](
    input_size=(IMG_WIDTH, IMG_HEIGHT, 3),
    num_classes=num_classes,
    trainable_layers=TRAINABLE_BASE_LAYERS,
    log_softmax=LOG_SOFTMAX,
    hidden_size=HIDDEN_SIZE
)

# Define callbacks
earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

mcp_save = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'best_weights'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)
reduce_lr_loss = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=7,
    verbose=1,
    epsilon=1e-4,
    mode='min'
)

# To show balanced accuracy
metrics = Metrics((testX, testY), model)

# compile model
print('Compiling model...')
opt = Adam(lr=LR, decay=LR / EPOCHS)
loss = (
    tf.keras.losses.CategoricalCrossentropy() if not LOG_SOFTMAX else (
        lambda labels, targets: tf.reduce_mean(
            tf.reduce_sum(
                -1 * tf.math.multiply(tf.cast(labels, tf.float32), targets),
                axis=1
            )
        )
    )
)



model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

print(f'Model has {model.count_params()} parameters')
print(f'Model summary {model.summary()}')

# train the head of the network
print('Starting training model...')
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss, metrics]
)

# make predictions on the testing set
print('Evaluating network...')
proba = model.predict(testX, batch_size=BATCH_SIZE)

# CSV: save predictions for inspection:
df = pd.DataFrame(proba)
df.to_csv(os.path.join(MODEL_DIR, "_preds_last_epoch.csv"))


# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(proba, axis=1)

print('classification report sklearn:')
print(
    classification_report(
        testY.argmax(axis=1), predIdxs, target_names=lb.classes_
    )
)

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
print('confusion matrix:')
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)

accuracy= balanced_accuracy_score(testY.argmax(axis=1), predIdxs)

# serialize the model to disk
print(f'Saving COVID-19 detector model on {MODEL_DIR} data...')
model.save(os.path.join(MODEL_DIR, 'last_epoch_8thMay'), save_format='h5')


#######################################################################ploting ################################
# plot the training loss and accuracy
N = EPOCHS
plt.style.use('ggplot')
plt.figure()
#plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
#plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy on COVID-19 Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(os.path.join(MODEL_DIR, 'loss.png'))

print('Done, shuttting down!')




import matplotlib.pyplot as plt
plt.plot(H.history["accuracy"])
plt.plot(H.history['val_accuracy'])
plt.plot(H.history['loss'])
#plt.plot(H.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","loss"])
plt.show()


def plot_confusion_matrix(cm, labels, save_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.axes
    df_cm = pd.DataFrame(
        cm,
        index=[i for i in ["COVID-19", "Pneumonia", "Healthy"]],
        columns=[i for i in ["COVID-19", "Pneumonia", "Healthy"]]
    )

    sn.set(font_scale=1.8)

    plt.xticks(
        np.arange(3) + 0.5, ("COVID-19", "Pneumonia", "Normal"),
        fontsize="18",
        va="center"
    )
    plt.yticks(
        np.arange(3) + 0.5, ("C", "P", "H"),
        rotation=0,
        fontsize="18",
        va="center"
    )
    
    
    # sn.heatmap(df_cm, annot=True, fmt="g", cmap="YlGnBu")
    sn.heatmap(df_cm, annot=True, fmt='', cmap="YlGnBu")
    # ax.xaxis.tick_bottom()
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=True
    )
    plt.xlabel("$\\bf{Predictions}$", fontsize=20)
    plt.ylabel("$\\bf{Ground\ truth}$", fontsize=20)
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)

###ROC Curve


def plot_multiclass_roc(model, X_test, y_test, n_classes, figsize=(17, 6)):
    
    y_score = model.predict(X_test, batch_size=BATCH_SIZE)
    
    #y_score = model.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
   

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    
    for i in range(len(n_classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] =   auc(fpr[i], tpr[i])
        
   

    # roc for each class
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    
    
    
    for i in range(len(n_classes)):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) ' % roc_auc[i] +  n_classes[i] )
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
    


plot_multiclass_roc(model, testX, testY.argmax(axis=1), n_classes =  ["COVID-19", "Pneumonia", "Healthy"], figsize=(16, 10))

## accuracy_score

print("Accuracy", accuracy_score(testY.argmax(axis=1), predIdxs))

## AUC_Curve

testy = testY
proba = model.predict(testX, batch_size=BATCH_SIZE)

for i in range(len(n_classes)):
    y_test_bin = np.int32(testy == i)
    y_score = proba[:,i]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score, pos_label=0)
    plt.subplot(2,2,i)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
