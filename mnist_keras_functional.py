from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

# training a convolutional network on mnist data set using Keras with Tensorflow backend

inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation="relu")(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)
model = Model(inputs=inputs, outputs=predictions)
model.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, validation_images, train_labels, validation_labels = train_test_split(
    train_images, train_labels, test_size=(1.0/6.0), stratify=train_labels
)

train_images_size = train_images.shape[0]
test_images_size = test_images.shape[0]
validation_images_size = validation_images.shape[0]

train_images = train_images.reshape((train_images_size, 28, 28, 1))
train_images = train_images.astype("float32") / 255.0
test_images = test_images.reshape((test_images_size, 28, 28, 1))
test_images = test_images.astype("float32") / 255.0
validation_images = validation_images.reshape((validation_images_size, 28, 28, 1))
validation_images = validation_images.astype("float32") / 255.0

train_labels = to_categorical(train_labels)
validation_labels = to_categorical(validation_labels)
test_labels = to_categorical(test_labels)

callbacks_list = [
    EarlyStopping(monitor="acc", patience=2),
    ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2),
    TensorBoard(log_dir="./log_dir", histogram_freq=1)
]

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images,
          train_labels,
          epochs=10,
          batch_size=64,
          validation_data=(validation_images, validation_labels),
          callbacks=callbacks_list)

test_loss, test_acc = model.evaluate(test_images, test_labels)
