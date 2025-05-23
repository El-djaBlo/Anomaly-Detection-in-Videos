import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Load the data
X_train = np.load('D:/Minor Project Ayan/data/X_train.npy', allow_pickle=True)
y_train = np.load('D:/Minor Project Ayan/data/y_train.npy', allow_pickle=True)

# Convert to proper dtype
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

print(f"X_train dtype: {X_train.dtype}, shape: {X_train.shape}")
print(f"y_train dtype: {y_train.dtype}, shape: {y_train.shape}")
print(f"First element in X_train: {X_train[0]}")
print(f"First element in y_train: {y_train[0]}")

# Define model architecture
model = Sequential([
    LSTM(128, input_shape=(32, 512), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# Callbacks
checkpoint_dir = "D:/Minor Project Ayan/models/"
os.makedirs(checkpoint_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    TensorBoard(
        log_dir=os.path.join(checkpoint_dir, 'logs'),
        histogram_freq=1
    )
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save(os.path.join(checkpoint_dir, 'final_model.keras'))
print("Model training complete and saved.")
