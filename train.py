import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as K
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Import model architectures
from model_binary import model as binary_model
from model_multiclass import model as multiclass_model

# Define metrics
METRICS = [
    K.metrics.AUC(name='auc'),
    K.metrics.Precision(name='precision'),
    K.metrics.Recall(name='recall'),
    K.metrics.F1Score(name='f1_score')
]

class TrainingConfig:
    """Training configuration"""
    def __init__(self, model_type='binary'):
        self.model_type = model_type
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.0001
        self.image_size = (256, 256)
        self.validation_split = 0.2
        self.early_stopping_patience = 10
        self.num_classes = 2 if model_type == 'binary' else 3
        
        # Directory paths
        self.data_dir = 'data'
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.checkpoint_dir = 'checkpoints'
        self.logs_dir = 'logs'
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


def create_data_generators(config):
    """Create training and validation data generators"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen, test_datagen


def load_data(config):
    """Load training and validation data"""
    
    train_datagen, val_datagen, test_datagen = create_data_generators(config)
    
    # Load training data
    if os.path.exists(config.train_dir):
        train_generator = train_datagen.flow_from_directory(
            config.train_dir,
            target_size=config.image_size,
            batch_size=config.batch_size,
            class_mode='categorical',
            classes=config.num_classes
        )
    else:
        print(f"Warning: Training directory not found at {config.train_dir}")
        train_generator = None
    
    # Load validation data
    if os.path.exists(config.val_dir):
        val_generator = val_datagen.flow_from_directory(
            config.val_dir,
            target_size=config.image_size,
            batch_size=config.batch_size,
            class_mode='categorical',
            classes=config.num_classes
        )
    else:
        print(f"Warning: Validation directory not found at {config.val_dir}")
        val_generator = None
    
    # Load test data
    if os.path.exists(config.test_dir):
        test_generator = test_datagen.flow_from_directory(
            config.test_dir,
            target_size=config.image_size,
            batch_size=config.batch_size,
            class_mode='categorical',
            classes=config.num_classes,
            shuffle=False
        )
    else:
        print(f"Warning: Test directory not found at {config.test_dir}")
        test_generator = None
    
    return train_generator, val_generator, test_generator


def setup_callbacks(config):
    """Setup training callbacks"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.model_type}_{timestamp}"
    
    callbacks = [
        # Early stopping
        K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        K.callbacks.ModelCheckpoint(
            os.path.join(config.checkpoint_dir, f'{model_name}_best.h5'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        K.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        K.callbacks.TensorBoard(
            log_dir=os.path.join(config.logs_dir, model_name),
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks, model_name


def train_model(config):
    """Train the model"""
    
    print(f"Starting training for {config.model_type} model...")
    print(f"Model type: {config.model_type}")
    print(f"Number of classes: {config.num_classes}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    
    # Load data
    train_generator, val_generator, test_generator = load_data(config)
    
    if train_generator is None or val_generator is None:
        print("Error: Cannot load training or validation data!")
        return None
    
    # Setup callbacks
    callbacks, model_name = setup_callbacks(config)
    
    # Get model
    if config.model_type == 'binary':
        model = binary_model
    else:
        model = multiclass_model
    
    # Display model summary
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test data
    if test_generator is not None:
        print("\nEvaluating on test data...")
        test_loss, test_auc, test_precision, test_recall, test_f1 = model.evaluate(test_generator)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Generate predictions
        y_pred_probs = model.predict(test_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_generator.classes
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                    target_names=list(test_generator.class_indices.keys())))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(test_generator.class_indices.keys()),
                    yticklabels=list(test_generator.class_indices.keys()))
        plt.title(f'Confusion Matrix - {config.model_type.capitalize()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(config.logs_dir, f'{model_name}_confusion_matrix.png'))
        print(f"Confusion matrix saved to {config.logs_dir}/{model_name}_confusion_matrix.png")
        plt.close()
    
    # Plot training history
    plot_training_history(history, config, model_name)
    
    # Save model
    model_path = os.path.join(config.checkpoint_dir, f'{model_name}_final.h5')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, history


def plot_training_history(history, config, model_name):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC
    axes[0, 1].plot(history.history['auc'], label='Training AUC')
    axes[0, 1].plot(history.history['val_auc'], label='Validation AUC')
    axes[0, 1].set_title('Model AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.logs_dir, f'{model_name}_history.png'))
    print(f"Training history plot saved to {config.logs_dir}/{model_name}_history.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train FA-Net model')
    parser.add_argument('--model-type', type=str, default='binary', 
                        choices=['binary', 'multiclass'],
                        help='Model type: binary or multiclass')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(model_type=args.model_type)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.data_dir = args.data_dir
    
    # Train model
    model, history = train_model(config)


if __name__ == '__main__':
    main()
