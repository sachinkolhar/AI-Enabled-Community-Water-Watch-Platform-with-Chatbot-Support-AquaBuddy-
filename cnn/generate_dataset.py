import os
import numpy as np
import cv2  # OpenCV
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Create base directory
base_dir = 'water_issues_dataset'
categories = ['pollution', 'scarcity', 'misuse']
os.makedirs(base_dir, exist_ok=True)

# 2. Create subdirectories for each class
for category in categories:
    os.makedirs(os.path.join(base_dir, category), exist_ok=True)

# 3. Function to create a random synthetic image
def create_random_image(label):
    img = np.ones((150, 150, 3), dtype=np.uint8) * 255  # Start with white background

    for _ in range(random.randint(5, 15)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        thickness = random.randint(1, 5)
        
        if label == 'pollution':
            # Draw random dirty spots
            center = (random.randint(0, 149), random.randint(0, 149))
            radius = random.randint(5, 30)
            cv2.circle(img, center, radius, color, -1)
        
        elif label == 'scarcity':
            # Draw cracked dry shapes
            start_point = (random.randint(0, 149), random.randint(0, 149))
            end_point = (random.randint(0, 149), random.randint(0, 149))
            cv2.line(img, start_point, end_point, color, thickness)
        
        elif label == 'misuse':
            # Draw chaotic mixed shapes
            if random.choice([True, False]):
                pt1 = (random.randint(0, 149), random.randint(0, 149))
                pt2 = (random.randint(0, 149), random.randint(0, 149))
                cv2.rectangle(img, pt1, pt2, color, -1)
            else:
                center = (random.randint(0, 149), random.randint(0, 149))
                radius = random.randint(5, 20)
                cv2.circle(img, center, radius, color, -1)
    
    return img

# 4. Generate images
for category in categories:
    for i in range(50):  # generate 50 images per category
        img = create_random_image(category)
        file_path = os.path.join(base_dir, category, f"{category}_{i}.png")
        cv2.imwrite(file_path, img)

print("✅ Synthetic dataset created successfully!")

# 5. Train a CNN model using the generated dataset
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build a CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("🚀 Training the model...")
history = model.fit(train_generator, validation_data=validation_generator, epochs=20)

# Save the model
model.save('D:/RVU/Sem_4/hackthon/IEEE RVU/waterflask/cnn/saved_model.h5')
model.save('D:/RVU/Sem_4/hackthon/IEEE RVU/waterflask/cnn/saved_model')
print("✅ Model trained and saved successfully!")

# Print training and validation accuracy
print("Training Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])
