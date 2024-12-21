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
    
def validate_model(detector, test_dir):
    #validate model performance
    results = {
        'true_positives': 0,   #AI images correctly
        'true_negatives': 0,   #Real images correctly
        'false_positives': 0,   #AI images incorrectly
        'false_negatives': 0,   #Real images incorrectly
    }

    #Testing images
    ai_dir = os.path.join(validation_dir, 'ai_generated')
    if os.path.exists(ai_dir):
        print("\nTesting AI images:\n")
        for img in os.listdir(ai_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                result = self.detect_ai_image(os.path.join(ai_dir, img))
                
                if result:
                    is_correct = result['is_ai_generated']
                    print(f"{img}: {'✓' if is_correct else '✗'} (Confidence: {result['confidence']:.2%})")
                    if is_correct:
                        results['true_positives'] += 1
                    else:
                        results['false_negatives'] += 1
                else:
                    results['false_positives'] += 1
    
    #Testing real images
    real_dir = os.path.join(validation_dir, 'real')
    if os.path.exists(real_dir):
        print("\nTesting real images:\n")
        for img in os.listdir(real_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                result = self.detect_ai_image(os.path.join(real_dir, img))
                
                if result:
                    is_correct = not result['is_ai_generated']
                    print(f"{img}: {'✓' if is_correct else '✗'} (Confidence: {result['confidence']:.2%})")
                    if is_correct:
                        results['true_negatives'] += 1
                    else:
                        results['false_positives'] += 1
                else:
                    results['false_negatives'] += 1
    
    #Calculate accuracy metrics
    total = sum(results.values())
    if total > 0:
        accuracy = (results['true_positives'] + results['true_negatives']) / total
        precision = results['true_positives'] / (results['true_positives'] + results['false_positives']) if (results['true_positives'] + results['false_positives']) > 0 else 0
        recall = results['true_positives'] / (results['true_positives'] + results['false_negatives']) if (results['true_positives'] + results['false_negatives']) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\nValidation Results:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1_score:.2%}")
        print("\nDetailed Results:")
        print(f"✓ True Positives (AI images correctly identified): {results['true_positives']}")
        print(f"✗ False Positives (Real images marked as AI): {results['false_positives']}")
        print(f"✓ True Negatives (Real images correctly identified): {results['true_negatives']}")
        print(f"✗ False Negatives (AI images marked as real): {results['false_negatives']}")
    
    return {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        },
        'results': results
    }
    

def main():
    # Create test directories if they don't exist
    test_dirs = ['test/ai_generated', 'test/real']
    for dir_path in test_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    detector = AIImageDetector()
    
    # Check if directories are empty
    if not any(os.listdir(os.path.join('test', d)) for d in ['ai_generated', 'real']):
        print("\nPlease add test images to the following directories:")
        print("- test/ai_generated/: Add AI-generated images here")
        print("- test/real/: Add real photographs here")
        return
    
    print("\nStarting AI Image Detection validation...")
    # Call the standalone validate_model function
    validate_model(detector, 'test')

if __name__ == "__main__":
    main()
    
    # Run the main function