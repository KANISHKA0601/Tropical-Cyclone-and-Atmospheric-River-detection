# Tropical-Cyclone-and-Atmospheric-River-detection
**ClimateNet India: Expert-level Dataset for Atmospheric River and Tropical Cyclone Classification Using U-Net Image Segmentation:**
**Introduction:**

Accurately identifying and tracking extreme weather events is essential for understanding their behavior in a changing climate. Traditional methods of weather pattern recognition often rely on manual heuristics, resulting in disparities and difficulties in reconciling outputs. Recent advancements in deep learning (DL) have shown promise in tackling such challenges in computer vision tasks. Inspired by this success, we introduce a DL-based semantic segmentation approach to detect and track atmospheric rivers (ARs) and tropical cyclones (TCs) using the ClimateNet dataset.

ClimateNet is an expert-annotated dataset capturing ARs and TCs in high-resolution climate model output. Leveraging this dataset, we employ a UNet-based DL model for pixel-level identification and segmentation of ARs and TCs. Our approach enables precise spatial and temporal tracking of these extreme weather events, providing valuable insights into their behavior.

**Study Area:**
The study area encompasses the region between 20° S and 30° N latitude, and 110° E to 40° E longitude, covering the Indian subcontinent and the Indian Ocean.

**Dataset:**
We utilized the ClimateNet Expert Level dataset, which comprises global rasters of 17 atmospheric variable features and one label. The label assigns values of 0 for background, 1 for tropical cyclones, and 2 for atmospheric rivers. To focus on our study area, we cropped the images accordingly and partitioned the dataset into three subsets: training (353 samples), testing (61 samples), and validation (45 samples).

**Tools and Methods:**
We conducted our analysis using the Google Colab platform for coding, leveraging the T4 GPU for efficient computation. Our model architecture and other processes were implemented using TensorFlow, a popular deep learning framework. For our study, we employed the UNet architecture, a convolutional neural network (CNN) known for its effectiveness in image segmentation tasks. UNet has been widely used in various image segmentation applications, offering high performance and accuracy.

**Methodology:**
We employed various libraries for data preprocessing, model development, and evaluation. These include xarray (xr), netCDF4 (nc), numpy (np), pandas (pd), glob, and TensorFlow (tf). For model development, we used TensorFlow’s Keras API and implemented the UNet architecture. Model evaluation utilized sklearn’s metrics and matplotlib for visualization. We set 956 as our random seed/state.
Our dataset is stored in netCDF format, so we converted it into numpy arrays and transposed it to fit the required image dimensions for TensorFlow. Additionally, the dataset initially had imbalanced classes across the images. To address this, we calculated the percentage of each class present in every image. We then applied a filter, ensuring that each image contained at least either an atmospheric river (AR) or a tropical cyclone (TC) covering more than 2% of the area. After filtering, we obtained 119, 13, and 16 images for the training, testing, and validation sets, respectively. To control mismatching of shape of input file we divide it with 32. The model utilizes one-hot encoding for data representation.

Model Development:

The UNet architecture consists of an encoder-decoder network designed for semantic segmentation tasks. The model takes an input shape of (192, 224, 16) and follows a series of convolutional and pooling layers for feature extraction. Batch normalization and dropout layers are incorporated for regularization. The decoder part of the network involves upsampling and concatenation operations to recover spatial information. The final layer employs a softmax activation function to output class probabilities. With a total of 23 convolutional layers, the UNet model effectively captures spatial dependencies and features for accurate segmentation.

