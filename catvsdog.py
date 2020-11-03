from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator


def create_model():
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_model():
    model = create_model()
    train_datagen = ImageDataGenerator(rescale=1.0/255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train = train_datagen.flow_from_directory('dataset/train/', class_mode='binary', batch_size=64, target_size=(150, 150))
    validation = val_datagen.flow_from_directory('dataset/validation/', class_mode='binary', batch_size=64, target_size=(150, 150))

    model = model.fit(train, validation_data=validation, epochs=30)
    return model


def plot_data(history):
    plt.subplot(2, 1, 1)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='red')

    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

history = fit_model()
plot_data(history)