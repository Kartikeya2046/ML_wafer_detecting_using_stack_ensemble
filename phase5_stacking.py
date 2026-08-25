import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

# Configure GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def FNN(lr=1e-4):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(18,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Loading base classifier outputs...")
    with open(r"d:\assignment college\VLSI_PROJECT\data\mfe_fnn_outputs.pkl", 'rb') as f:
        mfe_outputs = pickle.load(f)
        
    with open(r"d:\assignment college\VLSI_PROJECT\data\cnn_outputs.pkl", 'rb') as f:
        cnn_outputs = pickle.load(f)
        
    y_train = mfe_outputs['y_train']
    y_test = mfe_outputs['y_test']
    
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=9)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=9)
    
    # Concatenate the probability outputs from the base classifiers
    X_train_concat = np.concatenate([mfe_outputs['train_prob'], cnn_outputs['train_prob']], axis=1)
    X_test_concat = np.concatenate([mfe_outputs['test_prob'], cnn_outputs['test_prob']], axis=1)
    
    print(f"X_train_concat shape: {X_train_concat.shape}")
    print(f"X_test_concat shape: {X_test_concat.shape}")
    
    model = FNN()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    print("Training Stacking Meta-Learner (FNN)...")
    model.fit(
        X_train_concat, y_train_onehot,
        validation_split=0.2,
        epochs=1000,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Evaluating on Test Set...")
    y_pred_prob = model.predict(X_test_concat)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    report = classification_report(y_test, y_pred, digits=4)
    print("Classification Report (Stacking Ensemble):")
    print(report)
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"F1 Macro: {f1_macro:.4f}")
    
    # Save the model and the outputs for Phase 6
    os.makedirs(r"d:\assignment college\VLSI_PROJECT\models", exist_ok=True)
    model.save(r"d:\assignment college\VLSI_PROJECT\models\stacking_model.keras")
    
    with open(r"d:\assignment college\VLSI_PROJECT\data\stacking_outputs.pkl", 'wb') as f:
        pickle.dump({
            'test_prob': y_pred_prob,
            'y_test': y_test,
            'report': report,
            'f1_macro': f1_macro,
            'mfe_f1_macro': mfe_outputs['f1_macro'],
            'cnn_f1_macro': cnn_outputs['f1_macro'],
            'mfe_report': mfe_outputs['report'],
            'cnn_report': cnn_outputs['report']
        }, f)
        
    print("Phase 5 Completed!")

if __name__ == "__main__":
    main()
