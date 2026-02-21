import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Configuration ---
PROCESSED_DATA_PATH = "processed_data"
MODEL_SAVE_PATH = "voice_model.h5"
EMBEDDING_MODEL_PATH = "embedding_model.h5"
# --------------------

# =========================
# FEATURE NORMALIZATION
# =========================
def apply_cmvn(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1e-8
    return (features - mean) / std

# =========================
# LIGHT DATA AUGMENTATION
# =========================
def augment_features(features):
    augmented = [features]

    # mild noise
    noise = np.random.normal(0, 0.01, features.shape)
    augmented.append(features + noise)

    # mild scaling
    scale = 0.95 + np.random.rand() * 0.1
    augmented.append(features * scale)

    return augmented

# =========================
# LOAD DATA
# =========================
def load_data(processed_data_path, augment=True):
    all_features, all_labels = [], []
    max_len = 0

    speaker_dirs = sorted([
        d for d in os.listdir(processed_data_path)
        if os.path.isdir(os.path.join(processed_data_path, d))
    ])
    num_speakers = len(speaker_dirs)

    print("\n" + "="*60)
    print(f"LOADING DATA: Found {num_speakers} speakers")
    print("="*60 + "\n")

    speaker_to_label = {name: idx for idx, name in enumerate(speaker_dirs)}

    for spk in speaker_dirs:
        spk_path = os.path.join(processed_data_path, spk)
        label = speaker_to_label[spk]
        count = 0

        for fname in os.listdir(spk_path):
            if not fname.endswith(".npy"):
                continue

            feat = np.load(os.path.join(spk_path, fname))
            feat = apply_cmvn(feat)

            samples = augment_features(feat) if augment else [feat]

            for s in samples:
                all_features.append(s)
                all_labels.append(label)
                max_len = max(max_len, s.shape[0])
                count += 1

        print(f"  {spk} (Label {label}): {count} samples")

    print(f"\n  Total samples: {len(all_features)}")
    print(f"  Max sequence length: {max_len}\n")

    return all_features, all_labels, max_len, num_speakers, speaker_to_label

# =========================
# PREPARE DATA
# =========================
def prepare_data(features_list, labels_list, max_len):
    X = pad_sequences(features_list, maxlen=max_len, padding='post', dtype='float32')
    y = np.array(labels_list)

    print("="*60)
    print("PREPARED DATA")
    print("="*60)
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}\n")

    return X, y

# =========================
# BUILD MODEL
# =========================
def build_and_train_model(X, y, num_classes):

    y_cat = to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    input_shape = (X_train.shape[1], X_train.shape[2])

    # =========================
    # 🔥 FIXED ARCHITECTURE
    # =========================
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),

        Bidirectional(LSTM(32)),
        Dropout(0.2),

        # 🔑 VOICE EMBEDDING (linear!)
        Dense(128, activation='linear', name="embedding_layer"),

        # classifier head
        Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    print("="*60)
    print("STARTING TRAINING")
    print("="*60)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # =========================
    # SAVE CLASSIFIER
    # =========================
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 Classifier saved: {MODEL_SAVE_PATH}")

    # =========================
    # SAVE EMBEDDING MODEL
    # =========================
    embedding_model = Model(
        inputs=model.input,
        outputs=model.get_layer("embedding_layer").output
    )

    embedding_model.save(EMBEDDING_MODEL_PATH)
    print(f"💾 Embedding model saved: {EMBEDDING_MODEL_PATH}\n")

    return history

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("\n" + "="*60)
    print("VOICE BIOMETRIC TRAINING SYSTEM (FIXED)")
    print("="*60)

    features_list, labels_list, max_len, num_speakers, speaker_mapping = load_data(
        PROCESSED_DATA_PATH,
        augment=True
    )

    if features_list and num_speakers > 1:
        X, y = prepare_data(features_list, labels_list, max_len)
        build_and_train_model(X, y, num_speakers)

        np.save("speaker_mapping.npy", speaker_mapping)
        print("💾 Speaker mapping saved")
        print("="*60)
        print("TRAINING COMPLETE!")
        print("="*60 + "\n")
    else:
        print("❌ Need at least 2 speakers!")
