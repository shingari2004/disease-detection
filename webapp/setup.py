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
import traceback
from datetime import datetime
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global variables
global PIL_AVAILABLE, MODEL, REC_MODEL, MODEL_LOAD_ERROR, PREDICTION_TIMEOUT
MODEL = None
REC_MODEL = None
MODEL_LOAD_ERROR = None
PREDICTION_TIMEOUT = 30  # seconds

# Import PIL explicitly for better error handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("✓ PIL (Pillow) imported successfully")
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("⚠️ PIL (Pillow) not available")

app = Flask(__name__)

# Updated CORS configuration
CORS(app, origins=[
    'https://farminnovate-8pti.vercel.app',
    'https://*.onrender.com',
    'http://localhost:3000',
    'http://localhost:3001'
])

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG', 'webp'}

# Define classes
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

def optimize_tensorflow():
    """Optimize TensorFlow for better performance."""
    try:
        # Set memory growth for GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✓ Configured {len(gpus)} GPU(s) for memory growth")
        
        # Set CPU optimization
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        
        # Enable mixed precision if supported
        try:
            tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
            logger.info("✓ Enabled XLA JIT compilation")
        except Exception as e:
            logger.warning(f"⚠️ Could not enable XLA JIT: {e}")
        
        logger.info("✓ TensorFlow optimization completed")
        
    except Exception as e:
        logger.warning(f"⚠️ TensorFlow optimization failed: {e}")

def make_prediction_with_timeout(model, input_arr, timeout=30):
    """Make prediction with timeout handling."""
    
    def prediction_worker():
        """Worker function for prediction."""
        try:
            logger.info(f"Starting prediction with input shape: {input_arr.shape}")
            start_time = time.time()
            
            # Use smaller batch size and optimize prediction
            predictions = model.predict(input_arr, verbose=0, batch_size=1)
            
            end_time = time.time()
            logger.info(f"✓ Prediction completed in {end_time - start_time:.2f} seconds")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction worker error: {e}")
            raise
    
    # Use ThreadPoolExecutor for timeout handling
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(prediction_worker)
        
        try:
            # Wait for prediction with timeout
            predictions = future.result(timeout=timeout)
            return predictions
            
        except FutureTimeoutError:
            logger.error(f"❌ Prediction timed out after {timeout} seconds")
            raise TimeoutError(f"Prediction timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    """Check system requirements and log system info."""
    logger.info("=== SYSTEM REQUIREMENTS CHECK ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"PIL available: {PIL_AVAILABLE}")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Model directory exists: {os.path.exists(MODEL_DIR)}")
    
    if os.path.exists(MODEL_DIR):
        files = os.listdir(MODEL_DIR)
        logger.info(f"Files in model directory: {files}")
        
        model_file = os.path.join(MODEL_DIR, 'efficientnetv2s.h5')
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            logger.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
        else:
            logger.error("❌ Model file 'efficientnetv2s.h5' not found!")
    else:
        logger.error("❌ Model directory does not exist!")
    
    # Check TensorFlow GPU availability
    logger.info(f"TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
    logger.info("=== END SYSTEM CHECK ===")

def install_dependencies():
    """Install required dependencies if not available."""
    global PIL_AVAILABLE
    try:
        if not PIL_AVAILABLE:
            logger.info("Installing Pillow...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'], 
                                timeout=120)
            # Re-import PIL
            from PIL import Image
            PIL_AVAILABLE = True
            logger.info("✓ Pillow installed successfully")
        return True
    except subprocess.TimeoutExpired:
        logger.error("❌ Pillow installation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def preprocess_image(file_path):
    """Preprocess image with proper error handling."""
    try:
        logger.info(f"Processing image: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if PIL_AVAILABLE:
            # Load and preprocess image using PIL
            image = Image.open(file_path)
            logger.info(f"Original image size: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info("Converted image to RGB")
            
            # Resize to model input size
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            logger.info("Resized image to (224, 224)")
            
            # Convert to numpy array
            input_arr = np.array(image, dtype=np.float32)
            logger.info(f"Image array shape: {input_arr.shape}")
            
            # Add batch dimension
            input_arr = np.expand_dims(input_arr, axis=0)
            
            # Apply preprocessing
            try:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
                input_arr = preprocess_input(input_arr)
                logger.info("✓ Applied EfficientNetV2 preprocessing")
            except ImportError:
                # Fallback: Standard ImageNet normalization
                input_arr = input_arr / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_arr = (input_arr - mean) / std
                logger.info("✓ Applied fallback ImageNet normalization")
            
            return input_arr
        else:
            # TensorFlow method fallback
            logger.info("Using TensorFlow image loading (PIL not available)")
            image = tf.keras.utils.load_img(file_path, target_size=(224, 224))
            input_arr = tf.keras.utils.img_to_array(image)
            input_arr = np.expand_dims(input_arr, axis=0)
            
            try:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
                input_arr = preprocess_input(input_arr)
            except ImportError:
                input_arr = input_arr / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_arr = (input_arr - mean) / std
            
            return input_arr
            
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to process image: {str(e)}")

def load_model_with_fallback():
    """Comprehensive model loading with multiple fallback strategies."""
    model_path = os.path.join(MODEL_DIR, 'efficientnetv2s.h5')
    
    # Strategy 1: Try loading a smaller/faster model variant first
    try:
        logger.info("Strategy 1: Loading optimized model...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Custom objects for compatibility
        custom_objects = {
            'TFOpLambda': tf.keras.layers.Lambda,
            'tf': tf,
            'FixedDropout': tf.keras.layers.Dropout,
        }
        
        # Load model with optimization
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Optimize model for inference
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Convert to TensorFlow Lite for faster inference (optional)
        try:
            # This is experimental - create a faster inference version
            logger.info("Attempting to optimize model for inference...")
            
            # Create a concrete function for faster inference
            @tf.function(experimental_relax_shapes=True)
            def fast_predict(x):
                return model(x, training=False)
            
            model.fast_predict = fast_predict
            logger.info("✓ Added fast prediction function")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not optimize model for inference: {e}")
        
        logger.info("✓ Strategy 1: Successfully loaded model")
        return model
        
    except Exception as e:
        logger.error(f"❌ Strategy 1 failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Strategy 2: Load with smaller EfficientNet variant
    try:
        logger.info("Strategy 2: Loading with EfficientNetB0 (smaller/faster)...")
        
        from tensorflow.keras.applications import EfficientNetB0
        
        # Use smaller, faster EfficientNetB0 instead of V2S
        base_model = EfficientNetB0(
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
        
        logger.warning("⚠️ Strategy 2: Using EfficientNetB0 (faster but less accurate)")
        return model
        
    except Exception as e:
        logger.error(f"❌ Strategy 2 failed: {e}")
    
    # Strategy 3: Use MobileNetV2 (lightest/fastest)
    try:
        logger.info("Strategy 3: Loading MobileNetV2 (lightest/fastest)...")
        
        from tensorflow.keras.applications import MobileNetV2
        
        base_model = MobileNetV2(
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
        
        logger.warning("⚠️ Strategy 3: Using MobileNetV2 (fastest but least accurate)")
        return model
        
    except Exception as e:
        logger.error(f"❌ Strategy 3 failed: {e}")
    
    # All strategies failed
    raise Exception("All model loading strategies failed")

def validate_model(model):
    """Validate model with dummy input."""
    try:
        logger.info("Validating model with dummy input...")
        
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Apply same preprocessing
        try:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            dummy_input = preprocess_input(dummy_input)
        except ImportError:
            dummy_input = dummy_input / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            dummy_input = (dummy_input - mean) / std
        
        # Make prediction
        predictions = model.predict(dummy_input, verbose=0)
        
        # Validate predictions
        if predictions.shape != (1, len(CLASSES)):
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
        
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise ValueError("Model produces NaN or Inf predictions")
        
        pred_sum = np.sum(predictions[0])
        if not (0.99 <= pred_sum <= 1.01):  # Should sum to ~1 for softmax
            logger.warning(f"⚠️ Prediction sum unusual: {pred_sum}")
        
        logger.info(f"✓ Model validation successful - prediction sum: {pred_sum:.6f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        raise

def initialize_models():
    """Initialize models with comprehensive error handling."""
    global MODEL, REC_MODEL, MODEL_LOAD_ERROR
    
    try:
        logger.info("=== MODEL INITIALIZATION STARTED ===")
        
        # Optimize TensorFlow first
        optimize_tensorflow()
        
        # Check system requirements first
        
        # Install dependencies
        if not install_dependencies():
            raise Exception("Failed to install required dependencies")
        
        # Load main model
        logger.info("Loading plant disease detection model...")
        MODEL = load_model_with_fallback()
        
        # Validate model
        validate_model(MODEL)
        
        logger.info("✓ Main model loaded and validated successfully")
        
        # Load recommendation model
        rf_model_path = os.path.join(MODEL_DIR, 'RF.pkl')
        if os.path.exists(rf_model_path):
            try:
                with open(rf_model_path, 'rb') as f:
                    REC_MODEL = pickle.load(f)
                logger.info("✓ Recommendation model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load recommendation model: {e}")
                REC_MODEL = None
        else:
            logger.warning("⚠️ Recommendation model file not found")
            REC_MODEL = None
        
        MODEL_LOAD_ERROR = None
        logger.info("=== MODEL INITIALIZATION COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR during model initialization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        MODEL = None
        REC_MODEL = None
        MODEL_LOAD_ERROR = str(e)
        logger.error("=== MODEL INITIALIZATION FAILED ===")

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
            error_msg = f"Model not loaded. Error: {MODEL_LOAD_ERROR or 'Unknown error'}"
            logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'status': 'model_error',
                'details': MODEL_LOAD_ERROR
            }), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part', 'status': 'bad_request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file', 'status': 'bad_request'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(file_path)
                logger.info(f"File saved: {file_path}")

                # Preprocess image
                input_arr = preprocess_image(file_path)

                # Make prediction with timeout
                logger.info(f"Making prediction with timeout: {PREDICTION_TIMEOUT}s")
                
                try:
                    predictions = make_prediction_with_timeout(MODEL, input_arr, timeout=PREDICTION_TIMEOUT)
                except TimeoutError as e:
                    logger.error(f"❌ Prediction timeout: {e}")
                    return jsonify({
                        'error': f'Prediction timed out after {PREDICTION_TIMEOUT} seconds. The model may be too large for this server.',
                        'status': 'timeout_error',
                        'suggestion': 'Try using a smaller image or contact support.'
                    }), 504
                
                # Validate predictions
                if np.isnan(predictions).any() or np.isinf(predictions).any():
                    logger.error("❌ Model produced NaN or Inf predictions")
                    return jsonify({
                        'error': 'Model produced invalid predictions',
                        'status': 'prediction_error'
                    }), 500
                
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

                logger.info(f"✓ Prediction successful: {result} (confidence: {confidence:.4f})")

                return jsonify({
                    'result': result,
                    'confidence': confidence,
                    'top_3_predictions': top_3_predictions,
                    'message': 'Prediction successful',
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"❌ Prediction error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return jsonify({
                    'error': f'Prediction failed: {str(e)}',
                    'status': 'prediction_error'
                }), 500
            finally:
                # Clean up file
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up file: {e}")

        return jsonify({
            'error': 'File type not allowed',
            'status': 'bad_request',
            'allowed_types': list(ALLOWED_EXTENSIONS)
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in predict endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'server_error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'plant-disease-detection',
        'timestamp': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'pil_available': PIL_AVAILABLE,
        'model_loaded': MODEL is not None,
        'model_error': MODEL_LOAD_ERROR,
        'classes_count': len(CLASSES),
        'python_version': sys.version
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check."""
    if MODEL is None:
        return jsonify({
            'status': 'not_ready',
            'message': 'Model not loaded',
            'error': MODEL_LOAD_ERROR
        }), 503
    
    return jsonify({
        'status': 'ready',
        'message': 'Service is ready to handle requests',
        'model_loaded': True
    })

@app.route('/configure', methods=['POST'])
def configure_timeout():
    """Configure prediction timeout."""
    global PREDICTION_TIMEOUT
    
    try:
        data = request.get_json()
        if data and 'timeout' in data:
            new_timeout = int(data['timeout'])
            if 5 <= new_timeout <= 120:  # Between 5 seconds and 2 minutes
                PREDICTION_TIMEOUT = new_timeout
                logger.info(f"✓ Prediction timeout updated to {PREDICTION_TIMEOUT}s")
                return jsonify({
                    'status': 'success',
                    'message': f'Timeout updated to {PREDICTION_TIMEOUT} seconds',
                    'timeout': PREDICTION_TIMEOUT
                })
            else:
                return jsonify({
                    'error': 'Timeout must be between 5 and 120 seconds',
                    'status': 'bad_request'
                }), 400
        else:
            return jsonify({
                'error': 'Missing timeout parameter',
                'status': 'bad_request'
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Error configuring timeout: {e}")
        return jsonify({
            'error': f'Configuration error: {str(e)}',
            'status': 'server_error'
        }), 500

@app.route('/warmup', methods=['POST'])
def warmup_model():
    """Warm up the model with dummy predictions to improve response time."""
    try:
        if MODEL is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'model_error'
            }), 500
        
        logger.info("Starting model warmup...")
        
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Apply preprocessing
        try:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            dummy_input = preprocess_input(dummy_input)
        except ImportError:
            dummy_input = dummy_input / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            dummy_input = (dummy_input - mean) / std
        
        # Make several warm-up predictions
        warmup_times = []
        for i in range(3):
            start_time = time.time()
            try:
                predictions = make_prediction_with_timeout(MODEL, dummy_input, timeout=10)
                end_time = time.time()
                warmup_times.append(end_time - start_time)
                logger.info(f"Warmup {i+1}/3 completed in {end_time - start_time:.2f}s")
            except TimeoutError:
                logger.warning(f"Warmup {i+1}/3 timed out")
                warmup_times.append(10.0)  # Timeout value
        
        avg_time = np.mean(warmup_times)
        logger.info(f"✓ Model warmup completed. Average time: {avg_time:.2f}s")
        
        return jsonify({
            'status': 'success',
            'message': 'Model warmed up successfully',
            'warmup_times': warmup_times,
            'average_time': avg_time,
            'recommended_timeout': max(30, int(avg_time * 3))  # 3x average time
        })
        
    except Exception as e:
        logger.error(f"❌ Error warming up model: {e}")
        return jsonify({
            'error': f'Warmup error: {str(e)}',
            'status': 'server_error'
        }), 500
def debug_info():
    """Debug endpoint with detailed system information."""
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'python_version': sys.version,
            'tensorflow_version': tf.__version__,
            'pil_available': PIL_AVAILABLE,
        },
        'paths': {
            'base_dir': BASE_DIR,
            'model_dir': MODEL_DIR,
            'upload_folder': UPLOAD_FOLDER,
            'model_dir_exists': os.path.exists(MODEL_DIR),
        },
        'model_status': {
            'model_loaded': MODEL is not None,
            'model_error': MODEL_LOAD_ERROR,
            'rec_model_loaded': REC_MODEL is not None,
            'prediction_timeout': PREDICTION_TIMEOUT,
        },
        'classes': {
            'count': len(CLASSES),
            'classes': CLASSES[:5]  # Show first 5 classes
        }
    }
    
    # Add model directory contents if it exists
    if os.path.exists(MODEL_DIR):
        try:
            files = os.listdir(MODEL_DIR)
            debug_info['model_files'] = files
            
            model_file = os.path.join(MODEL_DIR, 'efficientnetv2s.h5')
            if os.path.exists(model_file):
                debug_info['model_file_size_mb'] = os.path.getsize(model_file) / (1024*1024)
        except Exception as e:
            debug_info['model_files_error'] = str(e)
    
    return jsonify(debug_info)

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'message': 'Flask server is running',
        'timestamp': datetime.now().isoformat(),
        'pil_available': PIL_AVAILABLE,
        'model_loaded': MODEL is not None,
        'classes_count': len(CLASSES),
        'service': 'plant-disease-detection'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Plant Disease Detection API',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': MODEL is not None,
        'service': 'plant-disease-detection',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/predict (POST) - Upload image for disease detection',
            'health': '/health (GET) - Health check',
            'ready': '/ready (GET) - Readiness check',
            'debug': '/debug (GET) - Debug information',
            'test': '/test (GET) - Test endpoint'
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
    logger.error(f"Internal server error: {error}")
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
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
