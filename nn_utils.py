import os
import time
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

class TimeStopping(tf.keras.callbacks.Callback):
    """
    Custom callback to stop training once a certain amount of time (in seconds) has passed.
    """
    def __init__(self, max_seconds=None):
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.max_seconds is None:
            return  # no time limit
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            print(f"Max training time of {self.max_seconds} seconds reached. Stopping training.")
            self.model.stop_training = True

def validate_input_data(input_values, target_values):
    """Validate and prepare input data"""
    if not input_values or not target_values:
        raise ValueError("Input or target values cannot be empty")
    
    X = np.array(input_values)
    y = np.array(target_values)
    
    # If 1D array, reshape to 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
        
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def train_neural_network(
    input_values,
    target_values,
    output_path,
    max_training_time=300,
    patience=5
):
    """
    Trains a simple feedforward neural network using provided input/target data and saves
    the model plus a use_util.py file that can be used to generate predictions.

    Parameters:
    -----------
    input_data : list of lists (or 2D array-like)
        The input features for training. Each sub-list is one training example’s features.
    target_data : list of lists (or 2D array-like)
        The target (label) for each training example. 
        Can be shape (num_samples,) or (num_samples, num_outputs).
    model_output_path : str
        Path to a folder where the trained model and `use_util.py` will be saved.
    max_training_time : float or None (optional)
        Maximum training time in seconds. If None, uses EarlyStopping only.
        If set, training will stop once this time is exceeded (or early stopping triggers).
    patience : int (optional)
        Number of epochs with no improvement after which training will be stopped 
        (used for EarlyStopping).
    """

    try:
        # Validate and prepare data
        X, y = validate_input_data(input_values, target_values)

        # Ensure model output path exists
        os.makedirs(output_path, exist_ok=True)

        # Build a simple feedforward neural network
        # You can adjust layer sizes, activation functions, etc. as needed
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(y.shape[1])  # output layer (same dimension as y)
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mse'   # mean squared error
        )

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='loss',     # or 'val_loss' if using validation data
                patience=patience,
                restore_best_weights=True
            ),
            TimeStopping(max_seconds=max_training_time)
        ]

        print("Starting training...\n")
        # Fit the model
        history = model.fit(
            X,
            y,
            epochs=1000,            # Arbitrary high epoch count—will stop early if needed
            verbose=1,              # Print training progress each epoch
            callbacks=callbacks,
            validation_split=0.0    # If you have validation data, set this or pass 'validation_data'
        )
        print("\nTraining complete.")

        # Save the trained model into the specified folder.
        # TensorFlow will create a SavedModel folder if you use model.save(...) with a directory path.
        saved_model_path = os.path.join(output_path, "trained_model")
        model.save(saved_model_path)

        print(f"Model saved to {saved_model_path}")

        # Create the "use_util.py" script for inference
        use_util_path = os.path.join(output_path, "use_util.py")
        with open(use_util_path, "w", encoding="utf-8") as f:
            f.write(f"""import os
import numpy as np
import tensorflow as tf

def use_model(input_list):
    \"""
    Loads the trained model from the local 'trained_model' folder
    and runs inference on the provided input_list.

    Parameters:
    -----------
    input_list : list of lists
        Each sub-list is one set of input features.

    Returns:
    --------
    A Python list containing the model's predictions.
    \"""
    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), "trained_model")
    model = tf.keras.models.load_model(model_path)

    # Convert input_list to np.array
    X = np.array(input_list, dtype=float)

    # Run predictions
    preds = model.predict(X)

    # Convert predictions to a plain Python list before returning
    return preds.tolist()
""")

        print(f"'use_util.py' created at {use_util_path}")

        # Print final training stats
        final_loss = history.history["loss"][-1]
        print(f"Final training loss: {final_loss:.6f}")
        if max_training_time:
            print(f"Training was time-limited to {max_training_time} seconds (or until early stopping).")
        else:
            print(f"Training ended via EarlyStopping after {len(history.history['loss'])} epochs.")

        print("\nAll done! You can now import `use_util.py` and call `use_model(...)` for predictions.")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def predict(input_list):
    """Make predictions with validation"""
    try:
        X = np.array(input_list)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "trained_model"))
        return model.predict(X).tolist()
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
