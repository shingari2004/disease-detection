from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import pickle
from flask_cors import CORS
import h5py
import logging
import sys
import subprocess
global PIL_AVAILABLE
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PIL explicitly for better error handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL (Pillow) not available. Installing...")

app = Flask(__name__)

# Updated CORS configuration for Render
CORS(app, origins=[
    'https://your-nextjs-app.vercel.app',
    'https://*.onrender.com',
    'http://localhost:3000',
    'http://localhost:3001'
])

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG', 'webp'}

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

def install_dependencies():
    """Install required dependencies if not available."""
    try:
        if not PIL_AVAILABLE:
            logger.info("Installing Pillow...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'])
            PIL_AVAILABLE = True
            logger.info("✓ Pillow installed successfully")
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")

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
            
            # Add batch dimension
            input_arr = np.expand_dims(input_arr, axis=0)
            
            # Apply EfficientNetV2 preprocessing
            try:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
                input_arr = preprocess_input(input_arr)
                logger.info("✓ Applied EfficientNetV2 preprocessing")
            except ImportError:
                # Fallback: EfficientNet typically uses ImageNet normalization
                input_arr = input_arr / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_arr = (input_arr - mean) / std
                logger.info("✓ Applied ImageNet normalization as fallback")
            
            return input_arr
        else:
            # TensorFlow method fallback
            image = tf.keras.utils.load_img(file_path, target_size=(224, 224))
            input_arr = tf.keras.utils.img_to_array(image)
            input_arr = np.expand_dims(input_arr, axis=0)
            
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            input_arr = preprocess_input(input_arr)
            
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
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Recompile the model
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

def create_model_and_load_weights():
    """Create model architecture and load weights separately."""
    try:
        from tensorflow.keras.applications import EfficientNetV2S
        
        logger.info("Creating EfficientNetV2S model architecture...")
        
        # Create base model
        base_model = EfficientNetV2S(
            weights='imagenet',
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
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Try to load custom weights
        model_path = os.path.join(MODEL_DIR, 'efficientnetv2s.h5')
        
        if os.path.exists(model_path):
            try:
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
        try:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            dummy_input = preprocess_input(dummy_input)
        except ImportError:
            # Fallback preprocessing
            dummy_input = dummy_input / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            dummy_input = (dummy_input - mean) / std
        
        # Make prediction
        predictions = model.predict(dummy_input, verbose=0)
        
        logger.info(f"Dummy prediction shape: {predictions.shape}")
        logger.info(f"Dummy prediction sum: {np.sum(predictions[0]):.6f}")
        
        # Check if predictions are valid
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise Exception("Model produces NaN or Inf predictions")
        
        logger.info("✓ Model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise

# Initialize model variable
MODEL = None
REC_MODEL = None

def initialize_models():
    """Initialize models with proper error handling."""
    global MODEL, REC_MODEL
    
    try:
        # Install dependencies first
        install_dependencies()
        
        logger.info("Loading models...")
        MODEL = load_model_comprehensive()
        
        # Validate model
        validate_model_predictions(MODEL)
        
        logger.info("Model loaded successfully")
        
        # Load recommendation model if it exists
        rf_model_path = os.path.join(MODEL_DIR, 'RF.pkl')
        if os.path.exists(rf_model_path):
            with open(rf_model_path, 'rb') as f:
                REC_MODEL = pickle.load(f)
            logger.info("✓ Recommendation model loaded")
        else:
            REC_MODEL = None
            logger.warning("⚠️ Recommendation model not found")
        
        logger.info("✓ All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Critical error initializing models: {e}")
        # Set MODEL to None to handle gracefully in endpoints
        MODEL = None
        REC_MODEL = None

# Create upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Check if model is loaded
        if MODEL is None:
            return jsonify({
                'error': 'Model not loaded properly. Please check server logs.',
                'status': 'model_error'
            }), 500
        
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
                
                # Validate predictions
                if np.isnan(predictions).any() or np.isinf(predictions).any():
                    logger.error("Model produced NaN or Inf predictions")
                    return jsonify({'error': 'Model produced invalid predictions'}), 500
                
                # Get prediction results
                prediction_index = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][prediction_index])
                result = CLASSES[prediction_index]
                
                # Get top 3 predictions
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                top_3_predictions = []
                for idx in top_3_indices:
                    top_3_predictions.append({
                        'class': CLASSES[idx],
                        'confidence': float(predictions[0][idx])
                    })

                # Clean up
                os.remove(file_path)

                return jsonify({
                    'result': result,
                    'confidence': confidence,
                    'top_3_predictions': top_3_predictions,
                    'message': 'Prediction successful',
                    'status': 'success'
                })

            except Exception as e:
                # Clean up
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.error(f"Prediction error: {e}")
                return jsonify({
                    'error': f'Prediction failed: {str(e)}',
                    'status': 'prediction_error'
                }), 500

        return jsonify({'error': 'File type not allowed'}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'server_error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render monitoring."""
    return jsonify({
        'status': 'healthy',
        'service': 'plant-disease-detection',
        'tensorflow_version': tf.__version__,
        'pil_available': PIL_AVAILABLE,
        'model_loaded': MODEL is not None,
        'classes_count': len(CLASSES),
        'python_version': sys.version
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check for Render deployment."""
    if MODEL is None:
        return jsonify({
            'status': 'not_ready',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'ready',
        'message': 'Service is ready to handle requests'
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'message': 'Flask server is running on Render',
        'pil_available': PIL_AVAILABLE,
        'model_loaded': MODEL is not None,
        'classes_count': len(CLASSES),
        'service': 'plant-disease-detection'
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Debug endpoint to get model information."""
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Test model with dummy input
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        try:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            dummy_input = preprocess_input(dummy_input)
        except ImportError:
            # Fallback preprocessing
            dummy_input = dummy_input / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            dummy_input = (dummy_input - mean) / std
        
        predictions = MODEL.predict(dummy_input, verbose=0)
        
        model_info = {
            'input_shape': str(MODEL.input_shape),
            'output_shape': str(MODEL.output_shape),
            'total_params': int(MODEL.count_params()),
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
        'message': 'Plant Disease Detection API - Running on Render',
        'status': 'running',
        'service': 'plant-disease-detection',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/predict (POST) - Upload image for disease detection',
            'health': '/health (GET) - Health check',
            'ready': '/ready (GET) - Readiness check',
            'test': '/test (GET) - Test endpoint',
            'model-info': '/model-info (GET) - Model information'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'not_found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'internal_error'
    }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large',
        'status': 'file_too_large'
    }), 413

# Initialize models when the app starts
initialize_models()

if __name__ == "__main__":
    # Configuration for Render deployment
    port = int(os.environ.get('PORT', 10000))  # Render uses port 10000 by default
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)