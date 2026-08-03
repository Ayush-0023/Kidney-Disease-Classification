import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

        # Load model
        self.model = load_model(
            os.path.join("model", "model.h5")
        )

    def predict(self):
        imagename = self.filename

        # Load and resize image
        test_image = image.load_img(
            imagename,
            target_size=(224, 224)
        )

        # Convert image to NumPy array
        test_image = image.img_to_array(test_image)

        # Same normalization used during training
        test_image = test_image / 255.0

        # Add batch dimension:
        # (224, 224, 3) -> (1, 224, 224, 3)
        test_image = np.expand_dims(test_image, axis=0)

        # Model returns probabilities for both classes
        probabilities = self.model.predict(test_image)

        print(probabilities)

        # Select class with highest probability
        result = np.argmax(probabilities, axis=1)

        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]