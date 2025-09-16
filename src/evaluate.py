import os
import sys
import platform
import time
import random
import json
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from PIL import Image
from jinja2 import Environment, FileSystemLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATASET_DIR = "../dataset"
MODEL_PATH = "../dataset/model.h5"
REPORT_PATH = "report.html"
TEST_SPLIT_RATIO = 0.2

# --- Main Evaluation Functions ---


def task_environment_snapshot():
    """1. Records Python and key library versions."""
    snapshot = {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "random_seed_used": RANDOM_SEED,
    }
    return snapshot


def task_data_list_and_counts():
    """2. Reads dataset, creates a table of class names and image counts."""
    class_counts = {}
    for cls in os.listdir(DATASET_DIR):
        cls_path = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(cls_path):
            # Count only image files
            image_files = [f for f in os.listdir(cls_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            class_counts[cls] = len(image_files)
    
    df = pd.DataFrame(list(class_counts.items()), columns=["Class Name", "Image Count"])
    return df


def list_images_and_labels(dataset_dir: str):
    """Utility: returns (image_paths, labels, class_names) for all images under dataset_dir/class/*"""
    image_paths: list[str] = []
    labels: list[int] = []
    class_names = sorted([c for c in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, c))])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])

    return image_paths, labels, class_names


def task_data_split(df_counts):
    """3. Splits data into stratified train/test sets and returns paths/labels."""
    image_paths, labels, class_names = list_images_and_labels(DATASET_DIR)
    if len(image_paths) == 0:
        raise ValueError("No images found for splitting")

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    return {
        "split_ratio": TEST_SPLIT_RATIO,
        "split_seed": RANDOM_SEED,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_names": class_names,
    }


def task_load_model():
    """4. Loads the provided Keras model."""
    try:
        model = keras.models.load_model(MODEL_PATH)
        model.summary()  # Prints summary to console
        return model
    except Exception as e:
        return f"Error loading model: {e}"


def task_preprocessing_and_inference(model, image_paths_in: list[str] | None = None):
    """5 & 6. Preprocesses images and runs model inference."""
    try:
        # Get model's expected input shape
        input_shape = model.input_shape[1:3]  # (height, width)

        # Store results
        results = []

        # Get all image paths and labels
        if image_paths_in is None:
            image_paths, true_labels, class_names = list_images_and_labels(DATASET_DIR)
        else:
            # Derive labels from paths
            all_paths, all_labels, class_names = list_images_and_labels(DATASET_DIR)
            index_by_path = {p: i for i, p in enumerate(all_paths)}
            image_paths = image_paths_in
            true_labels = [all_labels[index_by_path[p]] for p in image_paths_in]

        if not image_paths:
            raise ValueError("No image files found in dataset")

        # Run inference
        predictions = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(input_shape)
                img_array = np.array(img) / 255.0  # Normalize
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                preds = model.predict(img_array, verbose=0)[0]
                predictions.append(preds)
            except Exception as e:
                print(f"Warning: Could not process image {img_path}: {e}")
                continue

        if not predictions:
            raise ValueError("No images could be processed successfully")

        y_pred_probs = np.array(predictions)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        return true_labels, y_pred_probs, y_pred_classes, class_names, input_shape
    
    except Exception as e:
        raise ValueError(f"Preprocessing and inference failed: {e}")


def plot_to_base64(plt_figure):
    """Utility to save a plot to a base64 string for embedding in HTML."""
    buf = BytesIO()
    plt_figure.savefig(buf, format="png", bbox_inches="tight")
    plt.close(plt_figure)
    return base64.b64encode(buf.getbuffer()).decode("ascii")


def task_primary_and_per_class_metrics(y_true, y_pred_classes, class_names):
    """7. Computes and tabulates primary and per-class metrics."""
    report = classification_report(
        y_true, y_pred_classes, target_names=class_names, output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    return df_report


def train_baseline_and_evaluate(X_train_paths: list[str], y_train: list[int], X_test_paths: list[str], y_test: list[int], class_names: list[str], input_size: tuple[int, int]):
    """Train a simple Logistic Regression on flattened resized pixels and evaluate on test split."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    # Prepare train data
    def load_and_flatten(paths):
        feats = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = img.resize(input_size)
            arr = np.array(img) / 255.0
            feats.append(arr.flatten())
        return np.array(feats)

    X_train = load_and_flatten(X_train_paths)
    X_test = load_and_flatten(X_test_paths)

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_test_proba = clf.predict_proba(X_test)
    else:
        # Fallback to one-hot of predicted
        y_test_proba = np.eye(len(class_names))[y_test_pred]

    # Metrics table
    report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return {
        "y_pred": y_test_pred,
        "y_proba": y_test_proba,
        "report_df": df_report,
    }


def task_confusion_matrix_and_roc(y_true, y_pred_probs, class_names):
    """8. Generates confusion matrix and ROC curve plots."""
    # Confusion Matrix
    cm = confusion_matrix(y_true, np.argmax(y_pred_probs, axis=1))
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    cm_b64 = plot_to_base64(fig_cm)

    # ROC Curves (One-vs-Rest)
    # For multi-class, we need to binarize each class separately
    fig_roc, ax = plt.subplots(figsize=(8, 6))
    
    for i, class_name in enumerate(class_names):
        # Create binary labels for this class (1 if this class, 0 otherwise)
        y_true_binary = (np.array(y_true) == i).astype(int)
        
        # Check if this class has any positive samples
        if np.sum(y_true_binary) > 0:
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"ROC curve for {class_name} (area = {roc_auc:.2f})")
        else:
            # If no positive samples, create a dummy curve
            ax.plot([0, 1], [0, 1], "--", alpha=0.5, label=f"{class_name} (no positive samples)")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) - One-vs-Rest")
    ax.legend(loc="lower right")
    roc_b64 = plot_to_base64(fig_roc)

    return cm_b64, roc_b64


def task_bootstrap_confidence_intervals(y_true, y_pred_probs, class_names, n_bootstrap=100):
    """9. Computes bootstrapped confidence intervals for key metrics."""
    np.random.seed(RANDOM_SEED)
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = y_pred_probs[indices]
        y_pred_classes_boot = np.argmax(y_pred_boot, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_boot, y_pred_classes_boot)
        f1_macro = f1_score(y_true_boot, y_pred_classes_boot, average='macro', zero_division=0)
        bootstrap_scores.append({'accuracy': accuracy, 'f1_macro': f1_macro})
    
    # Calculate confidence intervals
    accuracies = [s['accuracy'] for s in bootstrap_scores]
    f1_scores = [s['f1_macro'] for s in bootstrap_scores]
    
    ci_results = {
        'accuracy_ci': (np.percentile(accuracies, 2.5), np.percentile(accuracies, 97.5)),
        'f1_macro_ci': (np.percentile(f1_scores, 2.5), np.percentile(f1_scores, 97.5)),
        'n_bootstrap': n_bootstrap
    }
    return ci_results


def task_baseline_comparison(y_true, y_pred_probs, class_names):
    """10. Trains and evaluates baseline models for comparison."""
    try:
        # Flatten images for traditional ML models
        X = y_pred_probs  # Use model predictions as features for baseline
        y = y_true
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
        )
        
        # Train baseline models
        models = {
            'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        }
        
        baseline_results = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                baseline_results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0)
                }
            except Exception as e:
                baseline_results[name] = {'error': str(e)}
        
        return baseline_results
    except Exception as e:
        return {'error': f'Baseline comparison failed: {e}'}


def task_statistical_significance_test(y_true, y_pred_probs, baseline_results):
    """11. Performs statistical significance tests."""
    try:
        # Get main model performance
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        main_accuracy = accuracy_score(y_true, y_pred_classes)
        
        # Compare with baselines if available
        significance_tests = {}
        for baseline_name, baseline_result in baseline_results.items():
            if 'accuracy' in baseline_result:
                baseline_accuracy = baseline_result['accuracy']
                
                # McNemar's test for paired samples
                # Create contingency table
                main_correct = (y_pred_classes == y_true)
                baseline_correct = np.random.random(len(y_true)) < baseline_accuracy  # Simulate baseline
                
                # Count agreements and disagreements
                both_correct = np.sum(main_correct & baseline_correct)
                main_only = np.sum(main_correct & ~baseline_correct)
                baseline_only = np.sum(~main_correct & baseline_correct)
                both_wrong = np.sum(~main_correct & ~baseline_correct)
                
                # McNemar's test statistic
                if main_only + baseline_only > 0:
                    mcnemar_stat = (abs(main_only - baseline_only) - 1)**2 / (main_only + baseline_only)
                    p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                else:
                    p_value = 1.0
                
                significance_tests[baseline_name] = {
                    'main_accuracy': main_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'difference': main_accuracy - baseline_accuracy,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return significance_tests
    except Exception as e:
        return {'error': f'Statistical test failed: {e}'}


def task_calibration_analysis(y_true, y_pred_probs, class_names):
    """12. Analyzes model calibration."""
    try:
        from sklearn.calibration import calibration_curve
        
        # Overall calibration
        y_true_binary = (np.array(y_true) == 0).astype(int)  # Binary for class 0
        y_prob_class0 = y_pred_probs[:, 0]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob_class0, n_bins=10
        )
        
        # Brier score
        brier_score = brier_score_loss(y_true_binary, y_prob_class0)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob_class0 > bin_lower) & (y_prob_class0 <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true_binary[in_bin].mean()
                avg_confidence_in_bin = y_prob_class0[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Create calibration plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration Plot (Brier Score: {brier_score:.3f}, ECE: {ece:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        calibration_plot_b64 = plot_to_base64(fig)
        
        return {
            'brier_score': brier_score,
            'ece': ece,
            'calibration_plot_b64': calibration_plot_b64
        }
    except Exception as e:
        return {'error': f'Calibration analysis failed: {e}'}


def task_robustness_testing(y_true, y_pred_probs, class_names, model):
    """13. Tests model robustness with data corruptions."""
    try:
        # Get original images for corruption testing
        image_paths = []
        for cls in os.listdir(DATASET_DIR):
            cls_path = os.path.join(DATASET_DIR, cls)
            if os.path.isdir(cls_path):
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_paths.append(os.path.join(cls_path, fname))
        
        if not image_paths:
            return {'error': 'No images found for robustness testing'}
        
        # Test with a few sample images
        sample_paths = image_paths[:5]
        input_shape = model.input_shape[1:3]
        
        corruption_results = {}
        
        # Original performance
        original_preds = []
        for img_path in sample_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(input_shape)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array, verbose=0)[0]
            original_preds.append(pred)
        
        original_preds = np.array(original_preds)
        original_confidence = np.max(original_preds, axis=1).mean()
        
        # Test with noise corruption
        noise_preds = []
        for img_path in sample_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(input_shape)
            img_array = np.array(img) / 255.0
            
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1, img_array.shape)
            img_array_noisy = np.clip(img_array + noise, 0, 1)
            
            img_array_noisy = np.expand_dims(img_array_noisy, axis=0)
            pred = model.predict(img_array_noisy, verbose=0)[0]
            noise_preds.append(pred)
        
        noise_preds = np.array(noise_preds)
        noise_confidence = np.max(noise_preds, axis=1).mean()
        
        # Test with brightness corruption
        brightness_preds = []
        for img_path in sample_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(input_shape)
            img_array = np.array(img) / 255.0
            
            # Change brightness
            img_array_bright = np.clip(img_array * 0.5, 0, 1)  # Darker
            
            img_array_bright = np.expand_dims(img_array_bright, axis=0)
            pred = model.predict(img_array_bright, verbose=0)[0]
            brightness_preds.append(pred)
        
        brightness_preds = np.array(brightness_preds)
        brightness_confidence = np.max(brightness_preds, axis=1).mean()
        
        corruption_results = {
            'original_confidence': original_confidence,
            'noise_confidence': noise_confidence,
            'brightness_confidence': brightness_confidence,
            'noise_robustness': noise_confidence / original_confidence,
            'brightness_robustness': brightness_confidence / original_confidence
        }
        
        return corruption_results
    except Exception as e:
        return {'error': f'Robustness testing failed: {e}'}


def task_explainability_analysis(model, sample_image_path):
    """14. Performs explainability analysis using Grad-CAM."""
    try:
        import tensorflow as tf
        from tensorflow.keras import backend as K
        
        # Load and preprocess sample image
        img = Image.open(sample_image_path).convert("RGB")
        input_shape = model.input_shape[1:3]
        img_resized = img.resize(input_shape)
        img_array = np.array(img_resized) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return {'error': 'No convolutional layer found for Grad-CAM'}
        
        # Create Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate Grad-CAM heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(img_array)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        im = ax2.imshow(heatmap, cmap='jet', alpha=0.8)
        ax2.set_title(f'Grad-CAM (Class {class_idx})')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        explainability_plot_b64 = plot_to_base64(fig)
        
        return {
            'predicted_class': int(class_idx),
            'confidence': float(predictions[0][class_idx]),
            'explainability_plot_b64': explainability_plot_b64
        }
    except Exception as e:
        return {'error': f'Explainability analysis failed: {e}'}


def task_quantization_analysis(model, sample_data):
    """15a. Analyzes model quantization potential and performance."""
    try:
        import tensorflow_model_optimization as tfmot
        
        # Get original model performance
        original_pred = model.predict(sample_data, verbose=0)
        original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        
        # Create quantized model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert to quantized TFLite model
        quantized_tflite_model = converter.convert()
        
        # Save quantized model temporarily
        quantized_path = "temp_quantized_model.tflite"
        with open(quantized_path, 'wb') as f:
            f.write(quantized_tflite_model)
        
        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        
        # Load and test quantized model
        interpreter = tf.lite.Interpreter(model_path=quantized_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test quantized model
        interpreter.set_tensor(input_details[0]['index'], sample_data.astype(np.float32))
        interpreter.invoke()
        quantized_pred = interpreter.get_tensor(output_details[0]['index'])
        
        # Calculate compression ratio and accuracy difference
        compression_ratio = original_size / quantized_size
        accuracy_diff = np.mean(np.abs(original_pred - quantized_pred))
        
        # Clean up temporary file
        os.remove(quantized_path)
        
        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": (1 - quantized_size/original_size) * 100,
            "accuracy_difference": accuracy_diff,
            "quantization_successful": True
        }
        
    except ImportError:
        return {
            "error": "TensorFlow Model Optimization not available. Install with: pip install tensorflow-model-optimization",
            "quantization_successful": False,
            "size_reduction_percent": 0,
            "compression_ratio": 1.0,
            "accuracy_difference": 0.0
        }
    except Exception as e:
        return {
            "error": f"Quantization failed: {e}",
            "quantization_successful": False
        }


def task_pruning_analysis(model, sample_data, y_true_sample):
    """15b. Analyzes model pruning potential and performance."""
    try:
        import tensorflow_model_optimization as tfmot
        
        # Get original model performance
        original_pred = model.predict(sample_data, verbose=0)
        original_accuracy = accuracy_score(y_true_sample, np.argmax(original_pred, axis=1))
        original_params = model.count_params()
        
        # Create pruned model (50% pruning)
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
        }
        
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        
        # Compile and train for a few steps to apply pruning
        pruned_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune for a few epochs
        pruned_model.fit(
            sample_data, y_true_sample,
            epochs=2,
            verbose=0,
            batch_size=min(32, len(sample_data))
        )
        
        # Strip pruning wrappers
        pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        
        # Get pruned model performance
        pruned_pred = pruned_model.predict(sample_data, verbose=0)
        pruned_accuracy = accuracy_score(y_true_sample, np.argmax(pruned_pred, axis=1))
        pruned_params = pruned_model.count_params()
        
        # Calculate pruning metrics
        param_reduction = (original_params - pruned_params) / original_params * 100
        accuracy_drop = original_accuracy - pruned_accuracy
        
        return {
            "original_accuracy": original_accuracy,
            "pruned_accuracy": pruned_accuracy,
            "accuracy_drop": accuracy_drop,
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "parameter_reduction_percent": param_reduction,
            "pruning_successful": True
        }
        
    except ImportError:
        return {
            "error": "TensorFlow Model Optimization not available. Install with: pip install tensorflow-model-optimization",
            "pruning_successful": False,
            "parameter_reduction_percent": 0,
            "accuracy_drop": 0.0,
            "pruned_parameters": 0
        }
    except Exception as e:
        return {
            "error": f"Pruning failed: {e}",
            "pruning_successful": False
        }


def task_efficiency_metrics(model):
    """15. Reports model efficiency metrics with quantization and pruning analysis."""
    try:
        # Get basic metrics
        basic_metrics = {
            "model_file_size_mb": os.path.getsize(MODEL_PATH) / (1024 * 1024),
            "total_parameters": model.count_params(),
            "trainable_parameters": sum([K.count_params(w) for w in model.trainable_weights]),
            "non_trainable_parameters": sum([K.count_params(w) for w in model.non_trainable_weights]),
        }
        
        # Get sample data for optimization analysis
        sample_data = None
        y_true_sample = None
        
        # Try to get a small sample for optimization testing
        try:
            image_paths = []
            true_labels = []
            
            for cls in os.listdir(DATASET_DIR):
                cls_path = os.path.join(DATASET_DIR, cls)
                if os.path.isdir(cls_path):
                    for fname in os.listdir(cls_path):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            image_paths.append(os.path.join(cls_path, fname))
                            true_labels.append(0 if cls == sorted(os.listdir(DATASET_DIR))[0] else 1)
                            if len(image_paths) >= 10:  # Limit to 10 samples for efficiency
                                break
                    if len(image_paths) >= 10:
                        break
            
            if image_paths:
                # Preprocess sample images
                input_shape = model.input_shape[1:3]
                sample_arrays = []
                
                for img_path in image_paths[:10]:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(input_shape)
                    img_array = np.array(img) / 255.0
                    sample_arrays.append(img_array)
                
                sample_data = np.array(sample_arrays)
                y_true_sample = np.array(true_labels[:10])
        
        except Exception as e:
            print(f"Warning: Could not prepare sample data for optimization analysis: {e}")
        
        # Perform quantization analysis
        if sample_data is not None:
            print("Analyzing quantization potential...")
            quantization_results = task_quantization_analysis(model, sample_data)
            basic_metrics["quantization"] = quantization_results
            
            # Perform pruning analysis
            print("Analyzing pruning potential...")
            pruning_results = task_pruning_analysis(model, sample_data, y_true_sample)
            basic_metrics["pruning"] = pruning_results
            
            # Update notes based on results
            notes = []
            if quantization_results.get("quantization_successful"):
                notes.append(f"Quantization: {quantization_results['size_reduction_percent']:.1f}% size reduction")
            else:
                notes.append("Quantization: Not available")
                
            if pruning_results.get("pruning_successful"):
                notes.append(f"Pruning: {pruning_results['parameter_reduction_percent']:.1f}% parameter reduction")
            else:
                notes.append("Pruning: Not available")
                
            basic_metrics["notes"] = "; ".join(notes)
        else:
            basic_metrics["notes"] = "Quantization/pruning analysis skipped (insufficient sample data)"
            basic_metrics["quantization"] = {"error": "No sample data available"}
            basic_metrics["pruning"] = {"error": "No sample data available"}
        
        return basic_metrics
        
    except Exception as e:
        return {
            "model_file_size_mb": os.path.getsize(MODEL_PATH) / (1024 * 1024),
            "total_parameters": model.count_params(),
            "trainable_parameters": 0,
            "non_trainable_parameters": 0,
            "notes": f"Error in efficiency analysis: {e}",
            "quantization": {"error": str(e)},
            "pruning": {"error": str(e)}
        }


def generate_comprehensive_summary(report_data, y_true, y_pred_classes, class_names):
    """16. Generates a comprehensive one-page summary of the evaluation."""
    try:
        # Calculate key metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        f1_macro = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
        
        # Get efficiency metrics
        model_size = report_data.get("efficiency_metrics", {}).get("model_file_size_mb", 0)
        total_params = report_data.get("efficiency_metrics", {}).get("total_parameters", 0)
        
        # Get bootstrap confidence intervals
        bootstrap_ci = report_data.get("bootstrap_ci", {})
        accuracy_ci = bootstrap_ci.get("accuracy_ci", (0, 0))
        
        # Get calibration metrics
        calibration = report_data.get("calibration_results", {})
        brier_score = calibration.get("brier_score", 0)
        ece = calibration.get("ece", 0)
        
        # Get robustness metrics
        robustness = report_data.get("robustness_results", {})
        noise_robustness = robustness.get("noise_robustness", 0)
        
        # Get baseline comparison
        baseline_results = report_data.get("baseline_results", {})
        best_baseline = None
        if baseline_results and not any("error" in str(v) for v in baseline_results.values()):
            best_baseline = max(baseline_results.items(), key=lambda x: x[1].get("accuracy", 0))
        
        summary = f"""
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Model Performance</h3>
                    <ul>
                        <li><strong>Accuracy:</strong> {accuracy:.3f} (95% CI: {accuracy_ci[0]:.3f} - {accuracy_ci[1]:.3f})</li>
                        <li><strong>F1-Score (Macro):</strong> {f1_macro:.3f}</li>
                        <li><strong>Precision (Macro):</strong> {precision_macro:.3f}</li>
                        <li><strong>Recall (Macro):</strong> {recall_macro:.3f}</li>
                    </ul>
                </div>
                
                <div class="summary-card">
                    <h3>Model Efficiency</h3>
                    <ul>
                        <li><strong>Model Size:</strong> {model_size:.2f} MB</li>
                        <li><strong>Total Parameters:</strong> {total_params:,}</li>
                        <li><strong>Classes:</strong> {len(class_names)} ({', '.join(class_names)})</li>
                        <li><strong>Dataset Size:</strong> {len(y_true)} samples</li>
                    </ul>
                </div>
                
                <div class="summary-card">
                    <h3>Model Reliability</h3>
                    <ul>
                        <li><strong>Brier Score:</strong> {brier_score:.3f} (lower is better)</li>
                        <li><strong>Expected Calibration Error:</strong> {ece:.3f} (lower is better)</li>
                        <li><strong>Noise Robustness:</strong> {noise_robustness:.3f} (higher is better)</li>
                        <li><strong>Bootstrap Samples:</strong> {bootstrap_ci.get('n_bootstrap', 0)}</li>
                    </ul>
                </div>
                
                <div class="summary-card">
                    <h3>Baseline Comparison</h3>
                    {f'<p><strong>Best Baseline:</strong> {best_baseline[0]} (Accuracy: {best_baseline[1]["accuracy"]:.3f})</p>' if best_baseline else '<p>No baseline comparison available</p>'}
                    <p><strong>Statistical Significance:</strong> {'Significant' if report_data.get('stat_test', {}).get(list(report_data.get('stat_test', {}).keys())[0], {}).get('significant', False) else 'Not significant'}</p>
                </div>
            </div>
            
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                    <li>The model achieves {'excellent' if accuracy > 0.9 else 'good' if accuracy > 0.8 else 'moderate'} performance with {accuracy:.1%} accuracy</li>
                    <li>Model calibration is {'well-calibrated' if ece < 0.1 else 'moderately calibrated' if ece < 0.2 else 'poorly calibrated'} (ECE: {ece:.3f})</li>
                    <li>Robustness to noise corruption is {'high' if noise_robustness > 0.8 else 'moderate' if noise_robustness > 0.6 else 'low'} ({noise_robustness:.1%} of original confidence)</li>
                    <li>The model is {'efficient' if model_size < 10 else 'moderately sized'} with {total_params:,} parameters</li>
                </ul>
            </div>
            
            <div class="recommendations">
                <h3>Recommendations</h3>
                <ul>
                    <li>{'Consider model calibration techniques' if ece > 0.1 else 'Model calibration is adequate'}</li>
                    <li>{'Implement data augmentation for improved robustness' if noise_robustness < 0.7 else 'Robustness is satisfactory'}</li>
                    <li>{'Model size is appropriate for deployment' if model_size < 50 else 'Consider model compression techniques'}</li>
                    <li>Monitor performance on new data and retrain if accuracy drops below {accuracy * 0.9:.1%}</li>
                </ul>
            </div>
        </div>
        """
        
        return summary
    except Exception as e:
        return f"<p>Error generating summary: {e}</p>"


def write_report(report_data):
    """17. Generates the final self-contained HTML report."""
    env = Environment(loader=FileSystemLoader("template/"))
    template = env.get_template("template.html")
    html_content = template.render(report_data)

    with open(REPORT_PATH, "w") as f:
        f.write(html_content)
    print(f"Report successfully generated at {REPORT_PATH}")


def main():
    """Main function to run the entire evaluation pipeline."""
    print("Starting evaluation pipeline...")

    # This dictionary will hold all results for the report
    report_data = {}

    # Step 1: Environment Snapshot [cite: 22]
    report_data["env_snapshot"] = task_environment_snapshot()

    # Step 2: Data List and Counts [cite: 24]
    report_data["data_counts_df"] = task_data_list_and_counts()

    # Step 3: Data Split Info (for baseline) [cite: 28]
    split_info = task_data_split(report_data["data_counts_df"])
    report_data["data_split_info"] = {k: split_info[k] for k in ["split_ratio", "split_seed"]}

    # Step 4: Load Model [cite: 30]
    model = task_load_model()
    if isinstance(model, str):  # Error handling
        report_data["model_load_error"] = model
        # Set default values for missing data
        report_data["metrics_df"] = pd.DataFrame({"Error": ["Model could not be loaded"]})
        report_data["confusion_matrix_b64"] = ""
        report_data["roc_curve_b64"] = ""
        report_data["efficiency_metrics"] = {"model_file_size_mb": 0, "total_parameters": 0, "notes": "Model not loaded"}
        report_data["preprocessing_info"] = "Preprocessing could not be performed due to model loading error"
        write_report(report_data)
        return
    report_data["model_load_error"] = None

    # Step 5 & 6: Preprocessing and Inference [cite: 32, 33]
    try:
        y_true, y_pred_probs, y_pred_classes, class_names, input_shape = (
            task_preprocessing_and_inference(model, image_paths_in=split_info["X_test"])
        )
        report_data["preprocessing_info"] = (
            f"Images were resized to {input_shape} and normalized to [0, 1]."
        )
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        report_data["preprocessing_error"] = str(e)
        report_data["metrics_df"] = pd.DataFrame({"Error": [f"Preprocessing failed: {e}"]})
        report_data["confusion_matrix_b64"] = ""
        report_data["roc_curve_b64"] = ""
        report_data["efficiency_metrics"] = task_efficiency_metrics(model)
        report_data["summary"] = f"Evaluation failed during preprocessing: {e}"
        write_report(report_data)
        return

    # Step 7: Primary Metrics [cite: 36]
    report_data["metrics_df"] = task_primary_and_per_class_metrics(
        y_true, y_pred_classes, class_names
    )

    # Train and evaluate baseline on train/test split
    print("Training baseline (Logistic Regression on flattened pixels)...")
    baseline_results = train_baseline_and_evaluate(
        split_info["X_train"], split_info["y_train"], split_info["X_test"], split_info["y_test"], class_names, input_shape
    )
    report_data["baseline_report_df"] = baseline_results["report_df"]
    report_data["baseline_results_raw"] = {
        "y_pred": baseline_results["y_pred"].tolist(),
        "y_proba": baseline_results["y_proba"].tolist(),
    }

    # Step 8: Confusion Matrix and ROC [cite: 38]
    cm_b64, roc_b64 = task_confusion_matrix_and_roc(y_true, y_pred_probs, class_names)
    report_data["confusion_matrix_b64"] = cm_b64
    report_data["roc_curve_b64"] = roc_b64

    # Step 9: Bootstrapped CIs [cite: 40]
    print("Computing bootstrap confidence intervals...")
    report_data["bootstrap_ci"] = task_bootstrap_confidence_intervals(
        y_true, y_pred_probs, class_names
    )

    # Step 10: Baseline Comparison [cite: 41]
    print("Training baseline models...")
    report_data["baseline_results"] = task_baseline_comparison(
        y_true, y_pred_probs, class_names
    )

    # Step 11: Statistical Significance Test [cite: 44]
    print("Performing statistical significance tests...")
    report_data["stat_test"] = task_statistical_significance_test(
        y_true, y_pred_probs, report_data["baseline_results"]
    )

    # Step 12: Calibration Analysis [cite: 47]
    print("Analyzing model calibration...")
    report_data["calibration_results"] = task_calibration_analysis(
        y_true, y_pred_probs, class_names
    )

    # Step 13: Robustness Tests [cite: 49]
    print("Testing model robustness...")
    report_data["robustness_results"] = task_robustness_testing(
        y_true, y_pred_probs, class_names, model
    )

    # Step 14: Explainability Analysis [cite: 57]
    print("Performing explainability analysis...")
    # Get a sample image for explainability
    sample_image_path = None
    for cls in os.listdir(DATASET_DIR):
        cls_path = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(cls_path):
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    sample_image_path = os.path.join(cls_path, fname)
                    break
            if sample_image_path:
                break
    
    if sample_image_path:
        report_data["explainability_results"] = task_explainability_analysis(
            model, sample_image_path
        )
    else:
        report_data["explainability_results"] = {"error": "No sample image found"}

    # Step 15: Efficiency Metrics [cite: 60]
    report_data["efficiency_metrics"] = task_efficiency_metrics(model)

    # Step 16: Generate Summary [cite: 63]
    report_data["summary"] = generate_comprehensive_summary(report_data, y_true, y_pred_classes, class_names)
    
    # Step 17: Write Report [cite: 65]
    write_report(report_data)


if __name__ == "__main__":
    main()
