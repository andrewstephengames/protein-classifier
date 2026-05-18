import gradio as gr
import tensorflow as tf
import numpy as np

print("Rebuilding model architecture...")
# 1. Rebuild the exact same "skeleton" used in Kaggle (without the Dual-GPU strategy)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights=None # We don't need Google's weights, we have yours!
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation="softmax") # 2 Output classes
])

# 2. Pass a dummy image through it once to physically build the layers in memory
print("Initializing layers...")
model(tf.zeros((1, 224, 224, 3)))

# 3. Load ONLY the weights from your file, bypassing the corrupted dual-GPU map
print("Loading saved weights...")
model.load_weights('high_protein_food_classifier.keras')
print("✅ Model loaded successfully!")


# --- Gradio Web App Logic ---

def classify_image(img):
    # Resize the incoming image to match MobileNetV2 dimensions
    img_array = tf.image.resize(img, (224, 224))
    # Add the batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Run the prediction
    prediction = model.predict(img_batch, verbose=0)[0]
    
    # Keras sorted your Kaggle folders alphabetically: 
    # [0] = high_protein, [1] = other_food
    return {"High Protein": float(prediction[0]), "Other Food": float(prediction[1])}

# Create the Web UI
interface = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2),
    title="High-Protein Food Classifier",
    description="Upload a photo of food to see if it's high in protein."
)

# Launch the server
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)