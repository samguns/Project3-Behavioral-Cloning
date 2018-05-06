import csv, argparse
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Cropping2D,\
    Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from PIL import Image


def load_samples(path):
    datas = []
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            datas.append(line)


    train_samples, validation_samples = train_test_split(datas, test_size=0.2)

    return train_samples, validation_samples


def generator(path, samples, BATCH_SIZE):
    num_samples = len(samples)
    while 1:
        shuffle(samples)

        for offset in range(0, num_samples, BATCH_SIZE):
            batch_samples = samples[offset:offset+BATCH_SIZE]

            images = []
            angles = []

            for batch_sample in batch_samples:
                name = path + '/IMG/' + batch_sample[0].split('/')[-1]
                center_image = Image.open(name)
                center_image = np.asarray(center_image)
                augment_image = np.fliplr(center_image)

                center_angle = float(batch_sample[3])
                images.append(center_image)
                images.append(augment_image)
                angles.append(center_angle)
                angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def nvidia_e2e_model():
    row, col, ch = 160, 320, 3

    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 24), (0, 0)), input_shape=(row, col, ch)))
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu"))
    # model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2)))
    # model.add(BatchNormalization(axis=3))
    # model.add(Activation("relu"))
    model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation="relu"))
    model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    return model


def train(path, EPOCHS, LR, BATCH_SIZE, w=None):
    train_samples, validation_samples = load_samples(path)
    train_generator = generator(path, train_samples, BATCH_SIZE)
    train_steps_per_epoch = 2 * len(train_samples) / BATCH_SIZE
    validation_generator = generator(path, validation_samples, BATCH_SIZE)
    validation_steps_per_epoch = 2 * len(validation_samples) / BATCH_SIZE
    model = nvidia_e2e_model()
    model.summary()
    tensorboardCB = TensorBoard(log_dir='./graph', batch_size=BATCH_SIZE)
    earlyStopCB = EarlyStopping('val_loss', patience=3)

    if w is not None:
        model.load_weights(w)

    optimizer = Adam(lr=LR)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit_generator(generator=train_generator, validation_data= validation_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=validation_steps_per_epoch,
                        epochs=EPOCHS, callbacks=[tensorboardCB, earlyStopCB])
    model.save_weights('weights.h5')
    model.save('model.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='./', dest='path',
                        help='Path of sample datas.')
    parser.add_argument('-e', type=int, default=2, dest='epochs',
                        help='Number of epochs.')
    parser.add_argument('-b', type=int, default=128, dest='batch_size',
                        help='Batch size.')
    parser.add_argument('-l', type=float, default=0.001, dest='learning_rate',
                        help='Learning rate.')
    parser.add_argument('-w', type=str, default=None, dest='weights',
                        help='Pre-trained weights.')

    args = parser.parse_args()
    try:
        epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        weights = args.weights
        path = args.path
        train(path, epochs, learning_rate, batch_size, weights)

    except argparse.ArgumentError:
        parser.print_help()