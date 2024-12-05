import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

class ImageRecognizer:
    def __init__(self):
        """
        Initialize the image recognition model.
        We're using ResNet50, a pre-trained model on ImageNet dataset.
        """
        # Load pre-trained ResNet50 model
        self.model = ResNet50(weights='imagenet')

    def recognize_image(self, image_path):
        """
        Recognize and classify the contents of an image.
        
        Args:
            image_path (str): Path to the image file to be recognized
        
        Returns:
            list: Top 3 predicted classes with their probabilities
        """
        try:
            # Load the image and resize it to 224x224 pixels (ResNet50 input size)
            img = image.load_img(image_path, target_size=(224, 224))
            
            # Convert image to numpy array
            img_array = image.img_to_array(img)
            
            # Expand dimensions to create a batch of size (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocess the image (normalize pixel values)
            preprocessed_img = preprocess_input(img_array)
            
            # Make predictions
            predictions = self.model.predict(preprocessed_img)
            
            # Decode and return top 3 predictions
            return decode_predictions(predictions, top=3)[0]
        
        except Exception as e:
            print(f"Error processing image: {e}")
            return []

def main():
    # Create an instance of the ImageRecognizer
    recognizer = ImageRecognizer()
    
    # Example usage
    image_path = 'path/to/your/image.jpg'
    
    # Recognize the image
    results = recognizer.recognize_image(image_path)
    
    # Print results
    print("Top predictions:")
    for prediction in results:
        class_name, description, probability = prediction
        print(f"{description} ({class_name}): {probability * 100:.2f}%")

if __name__ == "__main__":
    main()

# Note: Before running, install required libraries:
# pip install tensorflow pillow numpy