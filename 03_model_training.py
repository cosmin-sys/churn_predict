import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import json
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_processed_data():
    """Load the processed features"""
    print("Loading processed features...")
    df = pd.read_csv('data/processed/features.csv')
    return df

def prepare_data(df):
    """Prepare data for modeling"""
    
    # Separate features and target
    X = df.drop(['customer_id', 'churned'], axis=1)
    y = df['churned']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, X.columns.tolist()

def split_data(X, y, test_size=0.2, val_size=0.2):
    """Split data into train/validation/test sets"""
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler, handling NaNs and infinities"""
    
    # Replace inf values with NaN
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaNs with 0 (or use mean of column: X_train.fillna(X_train.mean()))
    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    
    # Optional: clip extreme values to avoid overflow
    X_train = X_train.clip(-1e6, 1e6)
    X_val = X_val.clip(-1e6, 1e6)
    X_test = X_test.clip(-1e6, 1e6)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_model(input_dim):
    """Build neural network model"""
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("Neural network model built:")
    model.summary()
    
    return model

def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced dataset"""
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"Class weights: {class_weight_dict}")
    
    return class_weight_dict

def train_model(model, X_train, y_train, X_val, y_val, class_weight):
    """Train the model"""
    
    print("Training model...")
    start_time = time.time()
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history, training_time

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model performance"""
    
    print("Evaluating model...")
    
    # Predictions
    train_pred = (model.predict(X_train) > 0.5).astype(int).flatten()
    val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    test_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Probabilities for AUC
    train_prob = model.predict(X_train).flatten()
    val_prob = model.predict(X_val).flatten()
    test_prob = model.predict(X_test).flatten()
    
    # Calculate metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, train_pred),
            'precision': precision_score(y_train, train_pred),
            'recall': recall_score(y_train, train_pred),
            'f1': f1_score(y_train, train_pred),
            'auc': roc_auc_score(y_train, train_prob)
        },
        'validation': {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred),
            'recall': recall_score(y_val, val_pred),
            'f1': f1_score(y_val, val_pred),
            'auc': roc_auc_score(y_val, val_prob)
        },
        'test': {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'auc': roc_auc_score(y_test, test_prob)
        }
    }
    
    # Print results
    for dataset, scores in metrics.items():
        print(f"\n{dataset.upper()} METRICS:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
    
    return metrics

def save_metrics(metrics, training_time, data_shapes):
    """Save metrics to JSON file"""
    
    # Add metadata
    metrics['metadata'] = {
        'training_time_seconds': training_time,
        'data_shapes': data_shapes,
        'model_type': 'neural_network',
        'framework': 'tensorflow',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save to file
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Metrics saved to artifacts/metrics.json")

def check_acceptance_criteria(metrics):
    """Check if model meets acceptance criteria"""
    
    test_metrics = metrics['test']
    
    print("\n=== ACCEPTANCE CRITERIA CHECK ===")
    
    criteria = [
        ('Accuracy ≥ 0.70', test_metrics['accuracy'] >= 0.70, test_metrics['accuracy']),
        ('AUC ≥ 0.75', test_metrics['auc'] >= 0.75, test_metrics['auc']),
    ]
    
    all_passed = True
    for criterion, passed, value in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{criterion}: {status} (Actual: {value:.4f})")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Status: {'✅ ALL CRITERIA MET' if all_passed else '❌ CRITERIA NOT MET'}")
    
    return all_passed

if __name__ == "__main__":
    # Load processed data
    df = load_processed_data()
    
    # Prepare data
    X, y, feature_columns = prepare_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Build model
    model = build_model(X_train_scaled.shape[1])
    
    # Calculate class weights
    class_weight = calculate_class_weights(y_train)
    
    # Train model
    model, history, training_time = train_model(
        model, X_train_scaled, y_train, X_val_scaled, y_val, class_weight
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    )
    
    # Save metrics
    data_shapes = {
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test)
    }
    save_metrics(metrics, training_time, data_shapes)
    
    # Check acceptance criteria
    check_acceptance_criteria(metrics)
    
    # Save model
    os.makedirs('artifacts/model', exist_ok=True)
    model.save('artifacts/model/churn_model.h5')
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    print("✅ Model saved to artifacts/model/churn_model.h5")
    print("✅ Metrics saved to artifacts/metrics.json")
    print("✅ Ready for deployment and testing")
