import json
import os

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

add_markdown("# Wafer Map Defect Classification Training Pipeline")

add_code(r"""
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# Configure GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)
        
os.makedirs(r"d:\assignment college\VLSI_PROJECT\models", exist_ok=True)
""")

add_markdown("## Phase 3: Handcrafted Feature (MFE) FNN Base Classifier")
add_code(r"""
print("Loading extracted MFE data and labels...")
with open(r"d:\assignment college\VLSI_PROJECT\data\X_MFE.pkl", 'rb') as f:
    X_train_mfe, X_test_mfe = pickle.load(f)
with open(r"d:\assignment college\VLSI_PROJECT\data\y.pkl", 'rb') as f:
    y_train, y_test = pickle.load(f)
    
print(f"X_train_mfe shape: {X_train_mfe.shape}, y_train shape: {y_train.shape}")
print(f"X_test_mfe shape: {X_test_mfe.shape}, y_test shape: {y_test.shape}")

# Cast to float32 to save RAM
X_train_mfe = X_train_mfe.astype(np.float32)
X_test_mfe = X_test_mfe.astype(np.float32)

# Standardize the features (zero mean, unit variance) as mentioned in paper
mean = np.mean(X_train_mfe, axis=0)
std = np.std(X_train_mfe, axis=0)
std[std == 0] = 1 # prevent division by zero

X_train_mfe_scaled = (X_train_mfe - mean) / std
X_test_mfe_scaled = (X_test_mfe - mean) / std

# Calculate class weights for imbalance handling
classes = np.unique(y_train)
class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights_array)}
print(f"Computed class weights: {class_weight_dict}")

y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=9)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=9)
""")

add_code(r"""
# Build FNN Architecture (Table 3 from paper)
mfe_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(59,)),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(9, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
mfe_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Training MFE-FNN Base Classifier...")
mfe_model.fit(
    X_train_mfe_scaled, y_train_onehot,
    validation_split=0.2,
    epochs=1000,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)
""")

add_code(r"""
print("Evaluating MFE FNN on Test Set...")
y_pred_prob_mfe = mfe_model.predict(X_test_mfe_scaled, batch_size=64)
y_pred_mfe = np.argmax(y_pred_prob_mfe, axis=1)

report_mfe = classification_report(y_test, y_pred_mfe, digits=4)
print("Classification Report (MFE+FNN):")
print(report_mfe)

f1_macro_mfe = f1_score(y_test, y_pred_mfe, average='macro')
print(f"F1 Macro: {f1_macro_mfe:.4f}")

# Generate predictions on train and test for the stacking ensemble
y_train_pred_prob_mfe = mfe_model.predict(X_train_mfe_scaled, batch_size=64)

mfe_model.save(r"d:\assignment college\VLSI_PROJECT\models\mfe_fnn_model.keras")

with open(r"d:\assignment college\VLSI_PROJECT\data\mfe_fnn_outputs.pkl", 'wb') as f:
    pickle.dump({
        'train_prob': y_train_pred_prob_mfe,
        'test_prob': y_pred_prob_mfe,
        'y_train': y_train,
        'y_test': y_test,
        'report': report_mfe,
        'f1_macro': f1_macro_mfe
    }, f)
print("Phase 3 MFE-FNN Completed!")
""")

add_markdown("## Phase 4: CNN Base Classifier")

add_code(r"""
print("Loading extracted CNN data and labels...")
with open(r"d:\assignment college\VLSI_PROJECT\data\X_CNN.pkl", 'rb') as f:
    X_train_cnn, X_test_cnn = pickle.load(f)
    
print(f"X_train_cnn shape: {X_train_cnn.shape}")

# Cast to float32, expand dims, and scale to save RAM
X_train_cnn = np.expand_dims(X_train_cnn.astype(np.float32), axis=-1)
X_test_cnn = np.expand_dims(X_test_cnn.astype(np.float32), axis=-1)

X_train_cnn = (X_train_cnn - 0.5) * 2.0
X_test_cnn = (X_test_cnn - 0.5) * 2.0

print(f"X_train_cnn pre-processed shape: {X_train_cnn.shape}")
""")

add_code(r"""
# Build VGGNet Architecture (Table 4 from paper)
cnn_model = tf.keras.Sequential([
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
cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Training CNN Base Classifier...")
cnn_model.fit(
    X_train_cnn, y_train_onehot,
    validation_split=0.2,
    epochs=1000,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)
""")

add_code(r"""
print("Evaluating CNN on Test Set...")
y_pred_prob_cnn = cnn_model.predict(X_test_cnn, batch_size=64)
y_pred_cnn = np.argmax(y_pred_prob_cnn, axis=1)

report_cnn = classification_report(y_test, y_pred_cnn, digits=4)
print("Classification Report (CNN):")
print(report_cnn)

f1_macro_cnn = f1_score(y_test, y_pred_cnn, average='macro')
print(f"F1 Macro: {f1_macro_cnn:.4f}")

# Generate predictions on train and test for the stacking ensemble
y_train_pred_prob_cnn = cnn_model.predict(X_train_cnn, batch_size=64)

cnn_model.save(r"d:\assignment college\VLSI_PROJECT\models\cnn_model.keras")

with open(r"d:\assignment college\VLSI_PROJECT\data\cnn_outputs.pkl", 'wb') as f:
    pickle.dump({
        'train_prob': y_train_pred_prob_cnn,
        'test_prob': y_pred_prob_cnn,
        'y_train': y_train,
        'y_test': y_test,
        'report': report_cnn,
        'f1_macro': f1_macro_cnn
    }, f)
print("Phase 4 CNN Completed!")
""")

add_markdown("## Phase 5: Stacking Ensemble")

add_code(r"""
print("Preparing Stacking Data...")
# We use the predicted probabilities from MFE and CNN as input features for the meta-learner
X_train_concat = np.concatenate([y_train_pred_prob_mfe, y_train_pred_prob_cnn], axis=1)
X_test_concat = np.concatenate([y_pred_prob_mfe, y_pred_prob_cnn], axis=1)

print(f"X_train_concat shape: {X_train_concat.shape}")
print(f"X_test_concat shape: {X_test_concat.shape}")

# FNN Meta-Learner
stacking_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(18,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])
stacking_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Training Stacking Meta-Learner...")
stacking_model.fit(
    X_train_concat, y_train_onehot,
    validation_split=0.2,
    epochs=1000,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)
""")

add_code(r"""
print("Evaluating Stacking Ensemble on Test Set...")
y_pred_prob_stack = stacking_model.predict(X_test_concat)
y_pred_stack = np.argmax(y_pred_prob_stack, axis=1)

report_stack = classification_report(y_test, y_pred_stack, digits=4)
print("Classification Report (Stacking Ensemble):")
print(report_stack)

f1_macro_stack = f1_score(y_test, y_pred_stack, average='macro')
print(f"F1 Macro: {f1_macro_stack:.4f}")

stacking_model.save(r"d:\assignment college\VLSI_PROJECT\models\stacking_model.keras")

with open(r"d:\assignment college\VLSI_PROJECT\data\stacking_outputs.pkl", 'wb') as f:
    pickle.dump({
        'test_prob': y_pred_prob_stack,
        'y_test': y_test,
        'report': report_stack,
        'f1_macro': f1_macro_stack,
        'mfe_f1_macro': f1_macro_mfe,
        'cnn_f1_macro': f1_macro_cnn,
    }, f)
print("Phase 5 Stacking Completed!")
""")

add_markdown("## Phase 6: Model Comparison")

add_code(r"""
target_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']

# Calculate per-class metrics
p_mfe, r_mfe, f1_mfe_class, _ = precision_recall_fscore_support(y_test, y_pred_mfe)
p_cnn, r_cnn, f1_cnn_class, _ = precision_recall_fscore_support(y_test, y_pred_cnn)
p_stk, r_stk, f1_stk_class, _ = precision_recall_fscore_support(y_test, y_pred_stack)

markdown_output = "# Final Model Comparison Results\\n\\n"
markdown_output += "## Per-Class F1 Score Comparison\\n\\n"
markdown_output += "| Class | MFE+FNN | CNN | Stacking Ensemble |\\n"
markdown_output += "|-------|---------|-----|-------------------|\\n"

for i, name in enumerate(target_names):
    markdown_output += f"| {name} | {f1_mfe_class[i]:.4f} | {f1_cnn_class[i]:.4f} | {f1_stk_class[i]:.4f} |\\n"
    
markdown_output += f"| **Macro Avg** | **{f1_macro_mfe:.4f}** | **{f1_macro_cnn:.4f}** | **{f1_macro_stack:.4f}** |\\n"

print(markdown_output)

with open(r"d:\assignment college\VLSI_PROJECT\results_comparison.md", 'w') as f:
    f.write(markdown_output)
    
print("Phase 6 comparison saved to results_comparison.md!")
""")

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"d:\assignment college\VLSI_PROJECT\training_pipeline.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Jupyter Notebook created successfully!")
