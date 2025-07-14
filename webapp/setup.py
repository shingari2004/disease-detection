from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import pickle
from flask_cors import CORS
import h5py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PIL explicitly for better error handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL (Pillow) not available. Please install: pip install Pillow")

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

# Define classes first (needed for model creation)
CLASSES = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry Powdery mildew', 'Cherry healthy',
    'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust',
    'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot',
    'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)',
    'Grape healthy', 'Orange Haunglongbing (Citrus greening)',
    'Peach Bacterial spot', 'Peach healthy', 'Pepper bell Bacterial spot',
    'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight',
    'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
    'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot',
    'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus',
    'Tomato healthy'
]

def preprocess_image(file_path):
    """Preprocess image with proper EfficientNet preprocessing."""
    try:
        logger.info(f"Processing image: {file_path}")
        
        if PIL_AVAILABLE:
            # Load and preprocess image
            image = Image.open(file_path)
            logger.info(f"Original image size: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            input_arr = np.array(image, dtype=np.float32)
            logger.info(f"Image array shape: {input_arr.shape}")
            logger.info(f"Image array min/max before preprocessing: {input_arr.min()}/{input_arr.max()}")
            
            # Add batch dimension
            input_arr = np.expand_dims(input_arr, axis=0)
            
            # CRITICAL: Use EfficientNetV2 preprocessing
            # EfficientNetV2 uses different preprocessing than simple /255 normalization
            try:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
                input_arr = preprocess_input(input_arr)
                logger.info("✓ Applied EfficientNetV2 preprocessing")
            except ImportError:
                # Fallback: EfficientNet typically uses ImageNet normalization
                # Convert to [0, 1] range first
                input_arr = input_arr / 255.0
                # Then apply ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_arr = (input_arr - mean) / std
                logger.info("✓ Applied ImageNet normalization as fallback")
            
            logger.info(f"Final input shape: {input_arr.shape}")
            logger.info(f"Final input min/max: {input_arr.min():.4f}/{input_arr.max():.4f}")
            
            return input_arr
        else:
            # TensorFlow method with proper preprocessing
            image = tf.keras.utils.load_img(file_path, target_size=(224, 224))
            input_arr = tf.keras.utils.img_to_array(image)
            input_arr = np.expand_dims(input_arr, axis=0)
            
            # Apply EfficientNetV2 preprocessing
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            input_arr = preprocess_input(input_arr)
            
            logger.info(f"TensorFlow preprocessing - Final shape: {input_arr.shape}")
            return input_arr
            
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise Exception(f"Failed to process image: {str(e)}")

def load_model_direct():
    """Try to load the model directly with proper handling."""
    try:
        model_path = os.path.join(MODEL_DIR, 'efficientnetv2s.h5')
        
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        
        # Define comprehensive custom objects
        custom_objects = {
            'TFOpLambda': tf.keras.layers.Lambda,
            'tf': tf,
            'tensorflow': tf,
            'FixedDropout': tf.keras.layers.Dropout,
            'EfficientNetV2S': tf.keras.applications.EfficientNetV2S,
        }
        
        # Try loading with custom objects
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            
            # CRITICAL: Recompile the model with proper configuration
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✓ Successfully loaded model directly")
            return model
            
        except Exception as e:
            logger.error(f"Direct loading failed: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Error in load_model_direct: {e}")
        raise

def create_model_and_load_weights():
    """Create model architecture and load weights separately."""
    try:
        from tensorflow.keras.applications import EfficientNetV2S
        
        logger.info("Creating EfficientNetV2S model architecture...")
        
        # Create base model
        base_model = EfficientNetV2S(
            weights='imagenet',  # Start with ImageNet weights
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Create full model
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax', name='predictions')
        ])
        
        # Compile first with ImageNet weights
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Try to load custom weights
        model_path = os.path.join(MODEL_DIR, 'efficientnetv2s.h5')
        
        if os.path.exists(model_path):
            try:
                # Method 1: Try to load full model weights
                logger.info("Attempting to load custom weights...")
                model.load_weights(model_path)
                logger.info("✓ Successfully loaded custom weights")
                
            except Exception as e:
                logger.warning(f"Could not load custom weights: {e}")
                logger.info("Using ImageNet weights as fallback")
        else:
            logger.warning("Custom model file not found, using ImageNet weights")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

def load_model_comprehensive():
    """Comprehensive model loading with proper error handling."""
    
    # Method 1: Try direct loading
    try:
        logger.info("Method 1: Attempting direct model loading...")
        return load_model_direct()
    except Exception as e:
        logger.error(f"Method 1 failed: {e}")
    
    # Method 2: Create model and load weights
    try:
        logger.info("Method 2: Creating model and loading weights...")
        return create_model_and_load_weights()
    except Exception as e:
        logger.error(f"Method 2 failed: {e}")
    
    # Method 3: Fallback to ImageNet weights only
    try:
        logger.info("Method 3: Using ImageNet weights only (fallback)...")
        from tensorflow.keras.applications import EfficientNetV2S
        
        base_model = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.warning("⚠️ Using ImageNet weights only - model not trained on plant diseases!")
        return model
        
    except Exception as e:
        logger.error(f"Method 3 failed: {e}")
        raise Exception("All model loading methods failed")

def validate_model_predictions(model):
    """Test model with dummy input to ensure it works properly."""
    try:
        logger.info("Validating model with dummy input...")
        
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Apply same preprocessing as real images
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        dummy_input = preprocess_input(dummy_input)
        
        # Make prediction
        predictions = model.predict(dummy_input, verbose=0)
        
        logger.info(f"Dummy prediction shape: {predictions.shape}")
        logger.info(f"Dummy prediction sum: {np.sum(predictions[0]):.6f}")
        logger.info(f"Dummy prediction contains NaN: {np.isnan(predictions).any()}")
        logger.info(f"Dummy prediction contains Inf: {np.isinf(predictions).any()}")
        
        # Check if predictions are valid
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise Exception("Model produces NaN or Inf predictions")
        
        if np.sum(predictions[0]) < 0.9 or np.sum(predictions[0]) > 1.1:
            logger.warning(f"Predictions don't sum to 1.0: {np.sum(predictions[0])}")
        
        logger.info("✓ Model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise

# Check if PIL is available
if not PIL_AVAILABLE:
    logger.info("Installing Pillow...")
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'])
        from PIL import Image
        PIL_AVAILABLE = True
        logger.info("✓ Pillow installed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to install Pillow: {e}")

# Initialize model variable
MODEL = None
REC_MODEL = None

# Load the models
try:
    logger.info("Loading models...")
    MODEL = load_model_comprehensive()
    
    # Validate model
    validate_model_predictions(MODEL)
    
    # Print model summary
    logger.info("Model summary:")
    MODEL.summary()
    
    # Load recommendation model if it exists
    rf_model_path = os.path.join(MODEL_DIR, 'RF.pkl')
    if os.path.exists(rf_model_path):
        with open(rf_model_path, 'rb') as f:
            REC_MODEL = pickle.load(f)
        logger.info("✓ Recommendation model loaded")
    else:
        REC_MODEL = None
        logger.warning("⚠️ Recommendation model not found")
    
    logger.info("✓ All models loaded successfully")
    
except Exception as e:
    logger.error(f"Critical error loading models: {e}")
    # Don't raise here - let the server start but handle in endpoints

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Check if model is loaded
        if MODEL is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Check if PIL is available
        if not PIL_AVAILABLE:
            return jsonify({'error': 'PIL (Pillow) library not available'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Preprocess image
                input_arr = preprocess_image(file_path)

                # Make prediction
                logger.info(f"Making prediction with input shape: {input_arr.shape}")
                predictions = MODEL.predict(input_arr, verbose=0)
                
                # Extensive debugging
                logger.info(f"Raw predictions shape: {predictions.shape}")
                logger.info(f"Raw predictions dtype: {predictions.dtype}")
                logger.info(f"Raw predictions sum: {np.sum(predictions[0]):.6f}")
                logger.info(f"Raw predictions sample: {predictions[0][:5]}")
                logger.info(f"Contains NaN: {np.isnan(predictions).any()}")
                logger.info(f"Contains Inf: {np.isinf(predictions).any()}")
                
                # Validate predictions
                if np.isnan(predictions).any() or np.isinf(predictions).any():
                    logger.error("Model produced NaN or Inf predictions")
                    return jsonify({'error': 'Model produced invalid predictions'}), 500
                
                # Check if predictions are all zeros or very small
                if np.max(predictions[0]) < 1e-10:
                    logger.warning("All predictions are very small - possible model issue")
                
                # Get prediction results
                prediction_index = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][prediction_index])
                result = CLASSES[prediction_index]
                
                logger.info(f"Predicted class: {result} (index: {prediction_index})")
                logger.info(f"Confidence: {confidence:.6f}")
                
                # Get top 3 predictions
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                top_3_predictions = []
                for idx in top_3_indices:
                    top_3_predictions.append({
                        'class': CLASSES[idx],
                        'confidence': float(predictions[0][idx])
                    })
                
                logger.info(f"Top 3 predictions: {top_3_predictions}")

                # Clean up
                os.remove(file_path)

                return jsonify({
                    'result': result,
                    'confidence': confidence,
                    'top_3_predictions': top_3_predictions,
                    'message': 'Prediction successful'
                })

            except Exception as e:
                # Clean up
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.error(f"Prediction error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

        return jsonify({'error': 'File type not allowed'}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'tensorflow_version': tf.__version__,
        'pil_available': PIL_AVAILABLE,
        'model_loaded': MODEL is not None,
        'classes_count': len(CLASSES)
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'message': 'Flask server is running',
        'pil_available': PIL_AVAILABLE,
        'model_loaded': MODEL is not None,
        'classes_count': len(CLASSES)
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Debug endpoint to get model information."""
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Test model with dummy input
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        dummy_input = preprocess_input(dummy_input)
        
        predictions = MODEL.predict(dummy_input, verbose=0)
        
        model_info = {
            'input_shape': MODEL.input_shape,
            'output_shape': MODEL.output_shape,
            'total_params': MODEL.count_params(),
            'layers': len(MODEL.layers),
            'compiled': MODEL.compiled_loss is not None,
            'dummy_prediction_sum': float(np.sum(predictions[0])),
            'dummy_prediction_max': float(np.max(predictions[0])),
            'dummy_prediction_contains_nan': bool(np.isnan(predictions).any()),
            'dummy_prediction_contains_inf': bool(np.isinf(predictions).any())
        }
        
        return jsonify(model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Plant Disease Detection API',
        'status': 'running',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'test': '/test (GET)',
            'model-info': '/model-info (GET)'
        }
    })

if __name__ == "__main__":
    # IMPORTANT: Modified for Heroku deployment
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)