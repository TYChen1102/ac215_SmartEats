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
│   ├── AC215_image_EfficientNet.ipynb     # fine-tuning of an EfficientNet model
│   ├── AC215_image_VGG_new.ipynb
│   ├── LLM-fintuning-Documentation.pdf    # Documentation of the LLM fine-tuning process
│   ├── frontpage.html                     # HTML file for application front page
│   ├── frontpage.jpg                      # screenshot of front page
│   ├── image_EDA.ipynb                    # EDA for image datasets
│   ├── LLM_RAG_preprocessing.ipynb        # EDA of data prepocessing for RAG raw data
│   └── predict_disease_ML.ipynb           # fine-tuning of a XGBClassifier model

├── references
├── reports
│   └── Statement of Work_Sample.pdf
└── src
    ├── llm-rag
    │   ├── docker-volumes/chromadb/
    │   ├── input-datasets/books/
    │   ├── output_RAG_different_config/
    │   ├── outputs
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── Readme
    │   ├── cli.py
    │   ├── docker-compose.yml
    │   ├── docker-entrypoint.sh
    │   └── docker-shell.sh
    ├── food-classification
    │   ├── secrets
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── food_model_EfficientNet.h5
    │   └── predict_food.py
    ├── food_to_nutrition
    │   ├── secrets
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   └── food_to_nutrition.py
    ├── gemini-finetuner
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── cli.py
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   └── transform_new.py
    ├── nutrition_predict_disease
    │   ├── models
    │   │   ├── Diabetes_model.pkl
    │   │   ├── High Cholesterol_model.pkl
    │   │   ├── Hypertension_model.pkl
    │   │   └── Obesity_model.pkl
    │   ├── secrets
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   └── nutrition_predict_disease.py
    ├── docker-compose.yml
    └── docker-shell.sh
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
   
4. nutrition_predict_disease: The nutrition_predict_disease container use the nutritional components and calories of the food item from food_to_nutrition container to predict the risk of four chronic diseases. 

   	**Input:** Nutritional information of the food item, and required secrets (provided via Docker)
   
   	**Output:** Predicted probabilities for the risk of developing diseases, including obesity, diabetes, hypertension, and high cholesterol

   
6. gemini-finetuner:
   
8. RAG_based_on_fine_tuned_model: Another container prepares data for the RAG model, including tasks such as chunking, embedding, and populating the vector database.
   src/RAG_on_fine_tuned_model/cli.py:  This script prepares the necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB).
        **Input:** Processed Raw data as txt. file, and user query text.
   	**Output:** Chunked data (.jsonl file), embedded data (.jsonl file), created ChromaDB instance, LLM response corresponding to the user query text, and LLM responses to our default evaluation questions (uploaded to GCP bucket as csv for different RAG configuration)
   

python cli.py --chunk --chunk_type char-split
python cli.py --chunk --chunk_type recursive-split

This will read each text file in the input-datasets/books directory; Split the text into chunks using the specified method (character-based or recursive);Save the chunks as JSONL files in the outputs directory

python cli.py --embed --chunk_type char-split
python cli.py --embed --chunk_type recursive-split

This will read the chunk files created in the previous section; Use Vertex AI's text embedding model to generate embeddings for each chunk; Save the chunks with their embeddings as new JSONL files.We use Vertex AI text-embedding-004 model to generate the embeddings

python cli.py --load --chunk_type char-split
python cli.py --load --chunk_type recursive-split

This will connect to your ChromaDB instance, create a new collection (or clears an existing one), and load the embeddings and associated metadata into the collection.

python cli.py --chat --query_text={your specific input}

This will generate the LLM response for specific user input using our vector database. Users could additionally specify “--chunk_type” to request two different vector base we generated using different split methods. 

python cli.py --process_questions --output-file={output file name}
This will run our evaluation queries based on different RAG configuration and upload the results to the GCP buckets.




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

**Data Versioning Strategy:**
We plan to incorporate a container for running DVC to keep track of the commits, logs, and different versions of datasets in later step. 

**Notebooks/Reports**
- Notebooks contains documentations and code that is not part of container: EDA, Application mockup, LLM fine-tuning documentation, ...
- Reports contains the project proposal submitted for Milestone 1.
- Results Data Folder for RAG:
  src/llm-rag/outputs: Output of chunking and data embedding
  src/llm-rag/output_RAG_different_contig: The log of LLM response with different RAG configuration
  src/llm-rag/input-datasets/books: The preprocessed text of our raw data.
  src/llm-rag/chromadb/: Data generated and used by Docker named chromadb, which is the container we used for setting up our vector database.

**Next Steps**
- Incorporate data versioning
- Combine some containers to reduce redundancy
  
