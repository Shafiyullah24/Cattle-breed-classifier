import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# === 1. Paths ===
base_dir = "IndianCattleBuffaloeBreeds-Dataset/breeds"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# === 2. Image settings ===
img_size = (224, 224)
batch_size = 32

# === 3. Data generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === 4. Load pretrained MobileNetV2 (Transfer Learning) ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # freeze base layers for faster training

# === 5. Build model ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 6. Train ===
print("\n🚀 Training started...")
history = model.fit(train_gen, validation_data=test_gen, epochs=10)
print("✅ Training completed!")

# === 7. Evaluate ===
loss, acc = model.evaluate(test_gen)
print(f"\n📊 Test Accuracy: {acc * 100:.2f}%")

# === 8. Save model ===
model.save("cattle_breed_classifier.h5")
print("💾 Model saved as 'cattle_breed_classifier.h5'")

# === 9. Optional: List class names ===
print("\n📂 Classes found:")
for i, cls in enumerate(train_gen.class_indices.keys()):
    print(f"{i}: {cls}")
