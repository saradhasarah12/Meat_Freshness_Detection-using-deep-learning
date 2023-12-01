# Meat_Freshness_Detection
Meat Freshness Detection Using Image Dataset.
The project aims to develop an image classification system that determines the freshness of meat. It uses deep learning techniques, particularly CNNs, to analyze images of meat and classify them into three categories: **'FRESH', 'HALF-FRESH', and 'SPOILED'**. 
The CNN model is trained to identify distinct visual patterns and features associated with different freshness states in meat products.

**Data Preprocessing:**
Image Normalization and Augmentation: The training images undergo normalization to scale pixel values between 0 and 1, ensuring consistent data representation. Image augmentation techniques, such as rotation, width and height shifting, shearing, zooming, and flipping, are applied to increase the dataset's diversity and robustness without changing the underlying labels.

**Model Architecture:**
Transfer Learning with Xception: Utilizing transfer learning, the Xception pre-trained model (with weights obtained from ImageNet) serves as the base model. This well-established architecture is used for feature extraction due to its capability in capturing intricate patterns within images.

**Customized Model Head:**
Additional layers are appended on top of the Xception base model. These include Batch Normalization, Global Average Pooling, Dense (fully connected) layers, Dropout, and the final Softmax layer. These layers are tailored to adapt the Xception model's output to suit the specific classification task of freshness prediction.
Compilation and Optimization: The model is compiled using the Adam optimizer with a specified learning rate and categorical cross-entropy loss function. The optimizer aims to minimize the loss function while updating the model's weights during training.
Callback for Training Control: A custom callback is implemented to monitor the model's performance during training. It checks for signs of overfitting, such as a large gap between training and validation accuracy, and stops training if necessary.

**Training and Evaluation:**
Model Training: The model is trained using the generated image batches from the training set for a specified number of epochs. The training process aims to optimize the model's parameters to make accurate predictions.
Model Evaluation: After training, the model's performance is assessed using a separate validation dataset. The prediction on a single test image shows the model's output probabilities for each freshness class, indicating the predicted freshness state of the meat based on the image features captured by the model.

**Project Outcome:**
Training Accuracy: 82.60%
Validation Accuracy: 90.47%
These accuracies indicate the performance of your model on the training and validation datasets, respectively, after 29 epochs of training. Your model seems to be learning well and generalizing effectively, as the validation accuracy is close to, or even slightly higher than, the training accuracy, suggesting minimal overfitting.

This performance is promising, indicating that your model has learned meaningful patterns from the training data and is capable of making reasonably accurate predictions on unseen validation data.
The project's ultimate goal is to provide a reliable system for assessing meat freshness through image analysis. By leveraging deep learning techniques, the model endeavors to accurately categorize meat images into their respective freshness states, enabling potential applications in food quality control and assurance.
