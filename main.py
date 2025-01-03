import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the metadata
metadata = pd.read_csv(r"D:\Secure Tech\AI for Healthcare\skin-cancer-mnist-ham10000\HAM10000_metadata.csv")
print(metadata.head())

# Add the .jpg extension to the image_id column if not present
metadata['image_id'] = metadata['image_id'].apply(lambda x: x + '.jpg')

# Check the first few rows after adding the extension
print(metadata['image_id'].head())

# Split the dataset into training and validation sets (80% train, 20% validation)
train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Set the directories where images are stored
image_directory_part_1 = r"D:\Secure Tech\AI for Healthcare\skin-cancer-mnist-ham10000\HAM10000_images_part_1"
image_directory_part_2 = r"D:\Secure Tech\AI for Healthcare\skin-cancer-mnist-ham10000\HAM10000_images_part_2"

# List both directories
image_directories = [image_directory_part_1, image_directory_part_2]

# Define image size and batch size
image_size = (224, 224)  # Resize all images to 224x224
batch_size = 32

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images
    rotation_range=40,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zooming in/out
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill pixels after transformations
)

# Validation data generator (just rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Function to find image path
def get_image_path(filename, directories=image_directories):
    for directory in directories:
        img_path = os.path.join(directory, filename)
        if os.path.exists(img_path):
            return img_path
    return None  # Return None if the image is not found

# Custom generator function to yield image and label pairs
def custom_flow_from_dataframe(dataframe, directories=image_directories, **kwargs):
    # Generate the file path for each image
    dataframe['image_path'] = dataframe['image_id'].apply(get_image_path)
    dataframe.dropna(subset=['image_path'], inplace=True)  # Remove rows with missing images
    return train_datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='image_path',  # Use the computed image paths
        y_col='dx',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        **kwargs
    )

# Set up the train data generator
train_generator = custom_flow_from_dataframe(
    train_df,
    directories=image_directories,
    shuffle=True
)

# Set up the validation data generator
val_generator = custom_flow_from_dataframe(
    val_df,
    directories=image_directories
)

# Check how many images are loaded and ensure no errors
print(f"Images in training set: {len(train_generator.filenames)}")
print(f"Images in validation set: {len(val_generator.filenames)}")



from tensorflow.keras import layers, models

# Build a simple CNN model
model = models.Sequential()

# Add convolutional layers with MaxPooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output of the convolutions
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(128, activation='relu'))

# Output layer (7 classes for skin cancer types)
model.add(layers.Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Print the model summary to see the architecture
model.summary()


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Number of batches per epoch
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=val_generator,
    validation_steps=len(val_generator),  # Number of batches in validation
)

# Save the model after training
model.save('skin_cancer_detection_model.h5')


import matplotlib.pyplot as plt

# Plot the training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_generator, steps=len(val_generator))
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
