# 🤖 Comprehensive Model Evaluation Framework

A robust, production-ready machine learning model evaluation framework that provides comprehensive analysis and beautiful HTML reports.

## ✨ Features

### 🎯 **Core Evaluation**

- **Environment Snapshot**: Python, TensorFlow, NumPy, Pandas versions
- **Dataset Analysis**: Class distribution and image counts
- **Model Loading**: Automatic Keras model loading with error handling
- **Preprocessing**: Image resizing, normalization, and batch processing
- **Inference**: Efficient model prediction on all dataset images

### 📊 **Performance Metrics**

- **Primary Metrics**: Accuracy, Precision, Recall, F1-Score (macro/micro/weighted)
- **Confusion Matrix**: Visual representation of classification results
- **ROC Curves**: One-vs-Rest ROC analysis with AUC scores
- **Bootstrap Confidence Intervals**: 95% CI for key metrics (100 bootstrap samples)

### 🏆 **Advanced Analysis**

- **Baseline Comparison**: Logistic Regression and Random Forest baselines
- **Statistical Significance**: McNemar's test for model comparison
- **Model Calibration**: Brier Score and Expected Calibration Error (ECE)
- **Robustness Testing**: Noise and brightness corruption analysis
- **Explainability**: Grad-CAM visualization for model interpretability

### ⚡ **Efficiency Metrics**

- **Model Size**: File size in MB
- **Parameter Count**: Total, trainable, and non-trainable parameters
- **Performance**: Inference time and memory usage

### 🎨 **Beautiful Reports**

- **Modern UI**: Responsive design with gradient backgrounds
- **Interactive Elements**: Hover effects and smooth transitions
- **Comprehensive Summary**: Executive summary with key findings
- **Visual Analytics**: Embedded plots and charts
- **Mobile Friendly**: Responsive design for all devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd gryffindor_submission
   ```

2. **Set up virtual environment**

   ```bash
   cd src
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Prepare your data**

   ```
   dataset/
   ├── classA/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── classB/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── model.h5
   ```

2. **Run evaluation**

   ```bash
   cd src
   source venv/bin/activate
   python evaluate.py
   ```

3. **View results**
   - Open `src/report.html` in your browser
   - Comprehensive analysis with visualizations

## 📁 Project Structure

```
gryffindor_submission/
├── dataset/                    # Dataset directory
│   ├── classA/                # Class A images
│   ├── classB/                # Class B images
│   └── model.h5               # Trained Keras model
├── src/                       # Source code
│   ├── evaluate.py            # Main evaluation script
│   ├── requirements.txt       # Python dependencies
│   ├── template/              # HTML template
│   │   └── template.html      # Report template
│   ├── venv/                  # Virtual environment
│   └── report.html            # Generated report
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

- `RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `DATASET_DIR`: Dataset directory path (default: "../dataset")
- `MODEL_PATH`: Model file path (default: "../dataset/model.h5")
- `REPORT_PATH`: Output report path (default: "report.html")

### Supported Image Formats

- PNG, JPG, JPEG, BMP, TIFF

### Model Requirements

- Keras/TensorFlow 2.x model saved as `.h5` file
- Compatible with image classification tasks
- Input shape: (batch_size, height, width, channels)

## 📊 Report Sections

### 1. **Executive Summary**

- Key performance metrics with confidence intervals
- Model efficiency and reliability scores
- Baseline comparison results
- Key findings and recommendations

### 2. **Environment & Configuration**

- Python and library versions
- Random seed information
- System configuration

### 3. **Dataset Overview**

- Class distribution table
- Image count per class
- Dataset statistics

### 4. **Model Information**

- Model loading status
- Architecture summary
- Preprocessing details

### 5. **Performance Metrics**

- Detailed classification report
- Per-class metrics
- Overall performance scores

### 6. **Bootstrap Analysis**

- 95% confidence intervals
- Statistical reliability
- Bootstrap sample information

### 7. **Baseline Comparison**

- Traditional ML model performance
- Statistical significance tests
- Performance comparison

### 8. **Model Calibration**

- Brier Score analysis
- Expected Calibration Error
- Calibration plots

### 9. **Robustness Testing**

- Noise corruption analysis
- Brightness variation testing
- Robustness scores

### 10. **Explainability**

- Grad-CAM visualizations
- Model interpretability
- Feature importance

### 11. **Visualizations**

- Confusion matrix
- ROC curves
- Calibration plots
- Grad-CAM heatmaps

### 12. **Efficiency Metrics**

- Model size and parameters
- Performance characteristics
- Deployment considerations

## 🛠️ Customization

### Adding New Metrics

```python
def custom_metric_function(y_true, y_pred):
    # Your custom metric implementation
    return metric_value

# Add to main evaluation pipeline
report_data["custom_metric"] = custom_metric_function(y_true, y_pred)
```

### Modifying Report Template

- Edit `src/template/template.html`
- Add new sections or modify styling
- Use Jinja2 templating syntax

### Extending Analysis

- Add new evaluation functions
- Implement additional robustness tests
- Include more explainability methods

## 🐛 Error Handling

The framework includes comprehensive error handling for:

- **Corrupted model files**: Graceful degradation with error messages
- **Empty datasets**: Clear error reporting
- **Missing images**: Skip problematic files with warnings
- **Template errors**: Fallback error pages
- **Memory issues**: Efficient batch processing

## 📈 Performance

- **Evaluation Time**: ~30-60 seconds for typical datasets
- **Memory Usage**: Optimized for large datasets
- **Report Size**: 100-200KB HTML with embedded visualizations
- **Scalability**: Handles datasets with thousands of images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the ML framework
- Scikit-learn for evaluation metrics
- Matplotlib/Seaborn for visualizations
- Jinja2 for templating

## 📞 Support

For questions or issues:

1. Check the error messages in the console output
2. Verify your dataset structure matches requirements
3. Ensure all dependencies are installed correctly
4. Check that your model is compatible with TensorFlow 2.x

---

**Made with ❤️ for comprehensive ML model evaluation**
# gryffindor_submissions
