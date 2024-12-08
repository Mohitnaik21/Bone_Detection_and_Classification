import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
# load the models
model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")

# categories for each result by index

#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["Elbow", "Hand", "Shoulder"]

#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']


# get image and model name, the default model is "Parts"
# Parts - bone type predict model of 3 classes
# otherwise - fracture predict for each part
def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

    # load image with 224px224p (the training model image size, rgb)
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)
    print(f"Raw prediction index: {prediction}")

    # chose the category and get the string prediction
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]
    print(f"Prediction result: {prediction_str}")

    return prediction_str


def lrp_heatmap(model, img_array):
    """
    Compute Layer-wise Relevance Propagation (LRP) heatmap.
    """
    # Convert the NumPy array to a TensorFlow tensor
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Compute gradients within a GradientTape context
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img_array)
        predictions = model(img_array)
        top_class = tf.argmax(predictions[0])

        # Get gradient of the top predicted class
        class_gradients = tape.gradient(predictions[:, top_class], img_array)
        print(f"Computed class gradients: {class_gradients}")
    # Compute relevance scores by summing across RGB channels
    heatmap = tf.reduce_sum(tf.abs(class_gradients[0]), axis=-1).numpy()

    # Normalize heatmap to [0, 1] range
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)
    print(f"Heatmap normalized with max value: {np.max(heatmap)}")
    return heatmap



def predict_with_lrp(img_path, model="Parts"):
    """
    Predict the class of an image and generate LRP heatmap.
    """
    size = 224
    if model == "Parts":
        chosen_model = model_parts
    else:
        chosen_model = {"Elbow": model_elbow_frac, "Hand": model_hand_frac, "Shoulder": model_shoulder_frac}[model]

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(size, size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Predict the class
    prediction = np.argmax(chosen_model.predict(img_array), axis=1)
    if model == "Parts":
        prediction_str = categories_parts[prediction.item()]
        print(f"Predicted body part: {prediction_str}")
        return prediction_str, None  # No heatmap needed for part prediction
    else:
        prediction_str = categories_fracture[prediction.item()]
        print(f"Predicted fracture status: {prediction_str}")

        # Generate heatmap
        heatmap = lrp_heatmap(chosen_model, img_array)

        # Save heatmap
        heatmap_dir = "heatmaps/"
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_path = os.path.join(heatmap_dir, f"{os.path.basename(img_path).split('.')[0]}_{model}_heatmap.png")
        plt.imshow(image.array_to_img(img_array[0]))  # Original image
        plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap
        plt.axis("off")
        plt.savefig(heatmap_path)
        plt.close()

        return prediction_str, heatmap_path
