About the Project:

The Banknote Authentication System is designed to address the growing threat of counterfeit currency by providing a reliable and accurate detection solution using machine learning techniques. The system classifies banknotes as genuine or counterfeit by analyzing four key statistical features extracted from banknote images, namely variance, skewness, kurtosis, and entropy. These features are processed using the XGBoost (Extreme Gradient Boosting) algorithm, chosen for its high accuracy, fast training speed, and ability to efficiently handle complex data patterns. Once trained, the model is deployed as an interactive web application using Streamlit, allowing users to input feature values and receive instant predictions on the authenticity of a banknote. By combining advanced machine learning with a simple and user-friendly interface, this project provides a practical and real-time solution for counterfeit currency detection, thereby contributing to enhanced financial security.

Dataset Origin:

The dataset used in this project consists of numerical features extracted from scanned images of banknotes, including both genuine and counterfeit samples to ensure balanced and accurate model training. High-resolution industrial scanners were used to capture fine visual details of the banknotes, after which Wavelet Transform techniques were applied to extract meaningful statistical features. These features effectively represent texture and frequency variations that differentiate authentic banknotes from counterfeit ones. The dataset is widely available from reliable sources such as the UCI Machine Learning Repository and Kaggle and is commonly used for academic research and machine learning experiments. Each data record represents a single banknote described by four extracted attributes—variance, skewness, kurtosis, and entropy—along with a class label indicating its authenticity.

Features (Independent Variables):

•	Variance: It provides the variation of pixel intensity.

•	Skewness: This represents the degree of asymmetry of the distribution of pixels.

•	Curtosis: This quantifies the pixel distribution as to how sharp it is or whether it is flat.

•	Entropy: It calculates the complexity or randomness of the image.

Target (Dependent Variable): 

•	Class: A binary variable indicating the type of banknote: 

▪ 0: Genuine banknote. 

▪ 1: Counterfeit banknote.


Dataset Link:

https://archive.ics.uci.edu/dataset/267/banknote+authentication
