# AC215 - Milestone2 - SmartEats

### Project:
In this project, we aim to develop SmartEats, an AI-powered application that analyzes food images, predicts disease risks based on dietary habits, and offers personalized nutrition advice. 
an application named SmartEats, which will take the food image as input from users and generate nutrition components with advise and potential disease risks. It will first require users to upload food pictures, and then it will use image recognition to estimate nutritional components like protein and fats and also display the amounts of each
component ideally. Based on the results, SmartEats will calculate calories, assess potential health risks, and provide tailored dietary suggestions, such as healthy recipes.

**Team Members:**
Jiayi Sun, Ninghui Hao, Qianwen Li, Taiyang Chen, Yantong Cui

**Group Name:**
The SmartEats Group

### Project Milestone 2 Organization

```
├── data 
├── notebooks
│   └── eda.ipynb
├── references
├── reports
│   └── Statement of Work_Sample.pdf
└── src
    ├── datapipeline
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── dataloader.py
    │   ├── docker-shell.sh
    │   ├── preprocess_cv.py
    │   ├── preprocess_rag.py
    ├── docker-compose.yml
    └── models
        ├── Dockerfile
        ├── docker-shell.sh
        ├── infer_model.py
        ├── model_rag.py
        └── train_model.py
```

## Milestone2 ###

In this milestone, we have the components for data management, including versioning, as well as the computer vision and language models.

**Data**
We upload our datasets to the bucket, allowing the entire group to access them. Within the notebook folder, there is an EDA file that helps introduce the dataset information in more details. 

### Data Pipeline Containers (src):

1. food-classification: The food-classification container recognizes the food in an image and stores the output back to Google Cloud Storage (GCS).

	**Input:** Image and required secrets (provided via Docker)

	**Output:** Name of the detected food name and predicted probability of the food

2. food_to_nutrition: The food_to_nutrition container links the food predicted from food_classification container & weight user inputs to the nutrient components and stores the output back to GCS.
   
   	**Input:** Food item predicted, the weight user inputs, and required secrets (provided via Docker)
   
   	**Output:** Nutrition components and calories of the food item identified
   
4. nutrition_predict_disease:
   
6. gemini-finetuner:
   
8. RAG_based_on_fine_tuned_model: Another container prepares data for the RAG model, including tasks such as chunking, embedding, and populating the vector database.


### Data Pipeline Overview:

1. **`src/food-classification/predict_food.py`**
   This script loads a fine-tuned EfficientNet model (food_model_EfficientNet.h5) and downloads a test food image from our bucket. Then it recognizes the food in the image and saves the food name as a JSON file to our bucket.

2. **`src/food-classification/Pipfile`**
   We used the following packages to help with food classification:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - opencv-python
   - tensorflow
   - keras
   - google-cloud-storage

4. **`src/preprocessing/Dockerfile`**
   Our Dockerfiles follow standard conventions.


## Data Pipeline Overview

1. **`src/datapipeline/preprocess_cv.py`**
   This script handles preprocessing on our 100GB dataset. It reduces the image sizes to 128x128 (a parameter that can be changed later) to enable faster iteration during processing. The preprocessed dataset is now reduced to 10GB and stored on GCS.

2. **`src/datapipeline/preprocess_rag.py`**
   This script prepares the necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB).

3. **`src/datapipeline/Pipfile`**
   We used the following packages to help with preprocessing:
   - `special cheese package`

4. **`src/preprocessing/Dockerfile(s)`**
   Our Dockerfiles follow standard conventions, with the exception of some specific modifications described in the Dockerfile/described below.


## Running Dockerfile
```
cd src                 # move into the directory with docker-shell.sh
sh docer-shell.sh      # the pipeline consisting of multiple containers will run sequentially 
```

**Models container**
- This container has scripts for model training, rag pipeline and inference
- Instructions for running the model container - `Instructions here`

**Notebooks/Reports**
This folder contains code that is not part of container - for e.g: Application mockup, EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations.

----
You may adjust this template as appropriate for your project.
