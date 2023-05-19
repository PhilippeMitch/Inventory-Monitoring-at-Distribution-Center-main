# Inventory Monitoring at Distribution Centers
More Distribution centers are adopting types of robotics technology as a part of their operations than ever before. Objects are carried in bins which can contain multiple objects. In this project, we build a model that can count the number of objects in a bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items. To build this project we used AWS SageMaker which is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. And machine learning engineering practices to fetch data from a database, upload the training data to an S3 bucket, and then train a machine learning model. 

# Project Set Up and Installation
In this section we will present the differents steps for our set up.
### 1. Set up Amazon SageMaker Studio domain

|   |  |
| ------------- | ------------- |
| On Amazon Sagemaker, clicked on Getting Started, then clicked on Set up Sagemaker Domain  | ![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/Set%20up%20Amazon%20SageMaker%20Studio%20domain1.png)  |
| In the Set up Sagemaker Domain page, typed a name for the Control Panel and select Create a new role  | ![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/Set%20up%20Amazon%20SageMaker%20Studio%20domain2.png)  |
| In the Create an IAM role pupop select Any S3 Bucket and clicked on Create role | ![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/Set%20up%20Amazon%20SageMaker%20Studio%20domain3.png) |
| Once the status is ready, launch the Studio domain by clicking on Launch app and select Studio | ![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/Set%20up%20Amazon%20SageMaker%20Studio%20domain4.png) |

# Dataset

### Overview
For this project we have used the Amazon Bin Image Dataset. The dataset contains 500,000 images of bins containing one or more objects. For each image there is a metadata file containing information about the image like the number of objects, its dimension and the type of object. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. However we only use a subset of this dataset which is about 10,436 images distributed into 5 classes.

![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/data_distribution.png)

### Access
Since the data is located in a S3 Bucket we used the json file provided to get the directory of images and the download_file methode to download the data. After downloaded the data we uploaded the data into our S3 Bucket.

# Model Training
For the training step we used different model. We used the resnet101 to train our model using the Adam optimizer with a learning_rate = 0.001 and a batch-size of 32. For our second training we used a model that we have created using the Adam optimizer and a learning_rate = 0.001.

# Machine Learning Pipeline
Our solution implements an Machine learning pipeline containing the following steps:
1. **Data ingestion**: In this step we retrieved the dataset, split the data into train, test and validation and then upload the data to a S3 Bucket.
2. **Model training script**: In this step we writed a script in what we defined some functions to process the data, initialized a pretrained model, train and test our model.
3. **Train in Sagemaker**: To train our model, we created a submission script so that we can submit a training job to an instance using a Jupyter Notebook. In the Jupyter Notebook we set up a training estimator and used the model training script to perform model profiling and debugging.

# Standout Suggestions

### Model Deployment
To deploy our model, we created a Pytorch model object and then call the deploy method from the PyTorch model object with the instance type with the parameter `instance_type` and the number of instances with the oarameter `initial_instance_count`.

![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/endpoint.png)

### Hyperparameter Tuning
For this experimentation i used the resnet101 pretrained model with three hyperparameters.

| Hyperparameter | Range |
| -------------- | ----- |
| Learning rate  | (0.001, 0.1) |
| Batch Size     | [8, 16, 32, 64] |
| Epochs         | (2, 10) |

After the training jobs here is the results.
![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/hyperparameter_tuning.png)

![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/best_model.png)

### Reduce Costs
To perform a cost analysis of the system, I create a Policy that allows access to the Cost Explorer API, attached this policy to the sagemaker role and used the Cost Explorer API to programmatically query your cost and usage data. To reduce the i used a spot instance by enable the `use_spot_instances` constructor arg in the PyToch model and set the `max_wait` constructor arg which is representing the amount of time we are willing to wait for Spot infrastructure to become available. 

### Multi-Instance Training
To perform the Multi-Instance training, we update the `instance_count` arg in the PyTorch nodel to three (3).

![This is an image](https://github.com/PedroToto/Inventory-Monitoring-at-Distribution-Center/blob/main/image/instance_count.png)
