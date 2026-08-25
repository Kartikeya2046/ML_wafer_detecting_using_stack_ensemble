import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Configure GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def main():
    print("Loading extracted CNN data and labels...")
    with open(r"d:\assignment college\VLSI_PROJECT\data\X_CNN.pkl", 'rb') as f:
        X_train, X_test = pickle.load(f)
    with open(r"d:\assignment college\VLSI_PROJECT\data\y.pkl", 'rb') as f:
        y_train, y_test = pickle.load(f)
        
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Cast to float32, expand dims, and scale to save RAM
    # We will let the GPU duplicate the channels via a Lambda layer to save 10GB+ of system RAM
    X_train = np.expand_dims(X_train.astype(np.float32), axis=-1)
    X_test = np.expand_dims(X_test.astype(np.float32), axis=-1)
    
    X_train = (X_train - 0.5) * 2.0
    X_test = (X_test - 0.5) * 2.0
    
    print(f"X_train pre-processed shape: {X_train.shape}")
    
    # Calculate class weights for imbalance handling
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {c: w for c, w in zip(classes, class_weights_array)}
    print(f"Computed class weights: {class_weight_dict}")
    
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=9)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=9)
    
    # Build VGGNet Architecture (Table 4 from paper)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64, 64, 1)),
        tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    print("Training CNN Base Classifier...")
    model.fit(
        X_train, y_train_onehot,
        validation_split=0.2,
        epochs=1000,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Evaluating on Test Set...")
    y_pred_prob = model.predict(X_test, batch_size=64)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    report = classification_report(y_test, y_pred, digits=4)
    print("Classification Report (CNN):")
    print(report)
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"F1 Macro: {f1_macro:.4f}")
    
    # Generate predictions on train and test for the stacking ensemble
    y_train_pred_prob = model.predict(X_train, batch_size=64)
    
    # Save the model and the outputs for Phase 5
    os.makedirs(r"d:\assignment college\VLSI_PROJECT\models", exist_ok=True)
    model.save(r"d:\assignment college\VLSI_PROJECT\models\cnn_model.keras")
    
    with open(r"d:\assignment college\VLSI_PROJECT\data\cnn_outputs.pkl", 'wb') as f:
        pickle.dump({
            'train_prob': y_train_pred_prob,
            'test_prob': y_pred_prob,
            'y_train': y_train,
            'y_test': y_test,
            'report': report,
            'f1_macro': f1_macro
        }, f)
        
    print("Phase 4 Completed!")

if __name__ == "__main__":
    main()
