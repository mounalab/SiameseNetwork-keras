
from livelossplot import PlotLossesKeras

from datetime import datetime
from time import time

import keras
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import Callback
from keras.callbacks import TensorBoard, CSVLogger, LearningRateScheduler
from keras.layers.merge import concatenate
from keras.layers import Input

from utils import euclidean_distance, eucl_dist_output_shape


class SiameseNetwork:
    """ Siamese Neural Network class
    """

    def __init__(self, encoder_model):

        # Set encoding parameters
        self.encoder_model = encoder_model

        # Get input shape from the encoder model
        self.input_shape = self.encoder_model.input_shape[1:]

        # Initialize siamese model
        self.siamese_model = None
        self.build()


    def build(self):
        """
        Initialize the siamese model structure using the input encoder network
        """

        # Define the tensors for the two input images
        left_input = Input(shape=self.input_shape, name="left_input")
        right_input = Input(shape=self.input_shape, name="right_input")

        # Generate the encodings (feature vectors) for the two inputs (left and right)
        encoded_l = self.encoder_model(left_input)
        encoded_r = self.encoder_model(right_input)


        # L2 distance layer between the two encoded outputs
        l2_distance_layer = Lambda(euclidean_distance,
                                   output_shape=eucl_dist_output_shape)

        l2_distance = l2_distance_layer([encoded_l, encoded_r])

        # Similarity measure prediction
        prediction = Dense(units=1)(l2_distance)

        self.siamese_model = Model(inputs=[left_input, right_input], outputs=prediction)


    def compile(self, *args, **kwargs):
        """
        Configures the model for training using the Keras model compile function
        """
        self.siamese_model.compile(*args, **kwargs)


    def fit(self,
        X_train_1,
        X_train_2,
        y_train,
        epochs=200,
        batch_size=32
        ):
        """
        Trains the model

        :param x_train_1: data points fed to the first sub-network
        :type 2-D Numpy array of float values
        :param x_train_2: data points fed to the second sub-network
        :type 2-D Numpy array of float values
        :param y_train: labels of each data points pair
        :type 2-D Numpy array of int values
        :param epochs: number of training epochs
        :type int
        :param batch_size: size of batches used at each forward/backward propagation
        :type int
        :return -
        :raises: -
        """

        ts = datetime.now().strftime('%d%m%Y_%H:%M')

        # This is used to save the best model, currently monitoring val_mape
        filepath = "checkpoint/Siamese.best"+ts+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Log file Path
        logfile = "log/"+ts+".log"

        #schedule = step_decay_schedule(initial_lr=1e-5, decay_factor=0.9, step_size=5)

        # Stop training if error does not improve within 20 iterations
        early_stopping_monitor = EarlyStopping(patience=20, restore_best_weights=True)

        #.... Siamese
        history_callback = self.siamese_model.fit([X_train_1, X_train_2], y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                                     verbose=1,
                                     callbacks=[#PlotLossesKeras(),
                                     early_stopping_monitor, checkpoint, CSVLogger(logfile)])
                                               #LRTensorBoard(log_dir="log/tb_log")])

    def restore(self, encoder_model):
        """
        Restore a previously trained siamese model

        :param encoder_model: encoding sub-network structure
        :type Keras Model
        :return: the trained encoding sub-model
        :rtype: Keras Model
        """

        # Load saved model
        self.siamese_model = load_model(checkpoint_path, compile=False)

        # Extract just the encoding sub model
        #encoder_model = trained_siamese_model.get_layer('sequential')
        model = encoder_model(self.siamese_model.output.shape)
        model.load_weights(checkpoint_path, by_name=True)


        input = Input(shape=self.input_shape)
        x = model(input)
        model = Model(input, x)

        return model
