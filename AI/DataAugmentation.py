import tensorflow as tf
import matplotlib.pyplot as plt

# Assuming you have an image tensor 'original_image' and an augmentation layer 'data_augmentation'
# Example:
original_image = tf.io.read_file("/content/3.jpg")
original_image = tf.image.decode_jpeg(original_image)
original_image = tf.image.resize(original_image, (150, 150))

# Define your data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.7),
    #tf.keras.layers.CenterCrop( 100, 100),
    #tf.keras.layers.RandomZoom(0.9),
])

# Apply augmentation
augmented_image = data_augmentation(original_image)

# Display original and augmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image.numpy().astype("uint8")) # Convert to NumPy array and uint8 for display
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Augmented Image")
plt.imshow(augmented_image.numpy().astype("uint8"))
plt.axis("off")
plt.show()
