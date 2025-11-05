import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")

# Load the dataset
paths = []
labels = []

# Update this path to your TESS dataset location
dataset_path = 'dataset/TESS'

for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            # Extract emotion label from filename
            label = filename.split('_')[-1].replace('.wav', '').lower()
            labels.append(label)

print(f"Total audio files loaded: {len(paths)}")

# Create dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels

print("\nEmotion distribution:")
print(df['label'].value_counts())

# Feature extraction with improved parameters
def extract_features(filename, duration=3, offset=0.5):
    """Extract MFCC, Chroma, and Mel Spectrogram features"""
    try:
        y, sr = librosa.load(filename, duration=duration, offset=offset)
        
        # MFCC features (40 coefficients)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        # Combine all features
        features = np.hstack([mfcc, chroma, mel])
        
        return features
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

print("\nExtracting features...")
X = []
y = []

for idx, row in df.iterrows():
    features = extract_features(row['speech'])
    if features is not None:
        X.append(features)
        y.append(row['label'])
    
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(df)} files")

X = np.array(X)
print(f"\nFeature shape: {X.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"Label classes: {label_encoder.classes_}")

# Save label encoder for later use
np.save('model/label_encoder_classes.npy', label_encoder.classes_)

# Reshape for LSTM input
X = np.expand_dims(X, axis=2)
print(f"Reshaped X: {X.shape}")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Build improved model
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(128, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Callbacks for better training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'model/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Save final model
model.save('model/emotion_model.h5')
print("\nModel saved to 'model/emotion_model.keras'")

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model/training_history.png')
print("\nTraining history plot saved to 'model/training_history.png'")
plt.show()

print("\n✓ Training complete!")