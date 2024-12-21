from locale import normalize
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

class AIImageDetector:
    def __init__(self):
        #use a pre-trained CNN model (ResNet50) as our feature extractor. It's good at extracting meaningful features from images
        #load pre-trained ResNet50 model without classification layers
        
        base_model = ResNet50(weights = 'imagenet', include_top = False)
        #Uses weights pre-trained on ImageNet dataset
        #"inlude_top = False" Removes the classification layers, we only want the feature extraction
        
        
        #Create model using the base model's input and output
        self.model = Model(inputs = base_model.input, outputs = base_model.layers[-1].output)
        
        #The idea is that AI-generated images often have different feature patterns compared to real images. ResNet50 will help us extract these features.
    
    def preprocess_image(self, image_path):
        """Preprocess the image for detection.This preprocessing is crucial because neural networks expect:       
        1) Consistent input sizes
        2) Normalized values
        3) Specific data formats (numpy arrays with batch dimensions)"""
        
        try:
            img = Image.open(image_path)
            img = img.resize((224, 224))    #Standard size for CNN models
            
            #Convert image to numpy array
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)   #Adds a batch dimension. Neural networks expect inputs in batches, even for single images
            
            #Normalize the pixel values from [-1 , 1]
            img_array = img_array.astype('float32') / 127.5 - 1
            
            return img_array
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def detect_ai_image(self, image_path):
        preprocessed_img = self.preprocess_image(image_path)
        
        if processed_img is None:
            return None
        
        #Extract features using our model
        features = self.model.predict(preprocessed_img)
        
        #Analyse feature patterns
        feature_std = np.std(features)      #Calculate standard deviation of features
        
        #Calculate confidence score
        #Lower feature variation → Higher confidence it's AI-generated
        #Higher feature variation → Lower confidence it's AI-generated
        normalized_std = (feature_std ) / 100.0
        confidence = 1.0 - min(normalized_std , 1.0)
        
        return confidence
    
def main():
        # Create test_images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        print("Created 'test_images' directory. Please add some test images.")            
        return
    
    detector = AIImageDetector()
        
        # Process all images in the test_images directory
    for image_file in os.listdir('test_images'):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join('test_images', image_file)
            result = detector.detect_ai_image(image_path)
                
            if result:
                print(f"\nResults for {image_file}:")
                print(f"AI Generated: {'Yes' if result['is_ai_generated'] else 'No'}")
                print(f"Confidence: {result['confidence']:.2%}")
    
if __name__ == "__main__":
    main()