# CNN Hyperparameter Study

This project explores how various **design choices and hyperparameters** affect the performance of **Convolutional Neural Networks (CNNs)** on small image classification tasks using:

- A 10-class subset of **CIFAR-100**
- The **Imagenette** dataset

## 📌 Objectives

We systematically investigate:

- Network **depth** and **width**
- Different **activation functions**: ReLU, LeakyReLU, Tanh, Sigmoid
- A variety of **optimizers** and **learning rates**
- The effect of **batch size** and **training duration**
- Trade-offs between learning speed, test accuracy, and overfitting

## 🧪 Experiments

All experiments are based on a simple three-block CNN architecture, progressively modified to measure the impact of each factor.

### 🛠️ Tools & Libraries Used

- **Python**
- **TensorFlow** and **Keras** – for building and training CNN models
- **NumPy** – for numerical operations
- **Pandas** – for data handling
- **Matplotlib** & **Seaborn** – for plotting and visualization
- **Scikit-learn** – for preprocessing and evaluation
- **OpenCV** – for image processing

## 📊 Results

The final report compares:

- Test accuracy
- Training curves
- Overfitting behavior
- Performance impact of architectural changes

## 📁 Files

- `cnn-hyperparameter-study.ipynb`: Jupyter Notebook with all experiments and analysis.

## 📚 Dataset Links

- **CIFAR-100 Subset**: A classic image classification dataset of 100 fine-grained classes. This project uses 10 randomly selected classes.  
  🔗 [Download from Keras](https://www.cs.toronto.edu/~kriz/cifar.html)

- **Imagenette**: A smaller, easier subset of ImageNet used for quick experimentation and benchmarking.  
  🔗 [Download from fast.ai](https://github.com/fastai/imagenette)


## 👩‍💻 Author

**Ayesha Sultana**  
📧 [mst.ayesha1702@gmail.com] 
- Course: *Introduction to Deep Learning* - 2025
- University of Turku, Finland
 


