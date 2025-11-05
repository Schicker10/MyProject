from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
from keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
print("Loading model...")
model = load_model('model/best_model.keras')
label_classes = np.load('model/label_encoder_classes.npy', allow_pickle=True)
print(f"Model loaded. Emotion classes: {label_classes}")

# Emotion descriptions for better presentation
emotion_descriptions = {
    'angry': '😠 Angry - The speaker sounds upset or frustrated',
    'disgust': '🤢 Disgust - The speaker expresses disgust or displeasure',
    'fear': '😨 Fear - The speaker sounds frightened or anxious',
    'happy': '😊 Happy - The speaker sounds joyful and cheerful',
    'neutral': '😐 Neutral - The speaker has a calm, neutral tone',
    'ps': '😮 Surprised - The speaker sounds pleasantly surprised',
    'sad': '😢 Sad - The speaker sounds sorrowful or down'
}

def extract_features(filename, duration=3, offset=0.5):
    """Extract audio features for prediction"""
    try:
        y, sr = librosa.load(filename, duration=duration, offset=offset)
        
        # MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        # Combine all features
        features = np.hstack([mfcc, chroma, mel])
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract features
            features = extract_features(filepath)
            
            if features is None:
                return jsonify({'error': 'Failed to process audio file'}), 500
            
            # Reshape for model input
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)
            
            # Make prediction
            predictions = model.predict(features, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_emotion = label_classes[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100
            
            # Get all probabilities
            all_probabilities = {}
            for idx, emotion in enumerate(label_classes):
                prob = float(predictions[0][idx]) * 100
                all_probabilities[emotion] = round(prob, 2)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'emotion': predicted_emotion,
                'confidence': round(confidence, 2),
                'description': emotion_descriptions.get(predicted_emotion, ''),
                'all_probabilities': all_probabilities
            })
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)