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
│   ├── AC215 - EDA.pdf                    # EDA pdf file 
│   ├── AC215_image_EfficientNet.ipynb     # train EfficientNet model
│   ├── AC215_image_VGG_new.ipynb          # train VGG model
│   ├── LLM-fintuning-Documentation.pdf    # Documentation of the LLM fine-tuning process
│   ├── LLM_RAG_preprocessing.ipynb        # construct the RAG vector database
│   ├── dataset3_EDA&preprocessing.ipynb   # EDA for nutrtion-disease dataset
│   ├── frontpage.html                     # HTML file for application front page
│   ├── frontpage.jpg                      # screenshot of front page
│   ├── image_EDA.ipynb                    # EDA for image datasets
│   └── predict_disease_ML.ipynb           # fine-tuning of a XGBClassifier model
├── references
├── reports
│   └── Statement of Work_Sample.pdf
└── src
    ├── food-classification
    │   ├── secrets
    │   ├── Dockerfile                     # To build the container for food classification
    │   ├── Pipfile                        # Define packages used in food classification
    │   ├── Pipfile.lock                   # Corresponding to Pipfile
    │   ├── food_model_EfficientNet.h5     # Fine-tuned EfficientNet model
    │   └── predict_food.py                # Loads a fine-tuned EfficientNet model, downloads data, recognizes the food and saves to our bucket.
    ├── food_to_nutrition
    │   ├── secrets                        
    │   ├── Dockerfile                     # build container to link nutrition components to food item
    │   ├── Pipfile                        # Define packages
    │   ├── Pipfile.lock
    │   └── food_to_nutrition.py           # Script to output nutrition components based on food predicted and weight input
    ├── gemini-finetuner
    │   ├── Dockerfile                     # Dockerfile to build container to train/fine-tune base LLM model
    │   ├── Pipfile                        # Define necessary packages and requirements
    │   ├── Pipfile.lock                   # Corresponding to Pipfile
    │   ├── cli.py                         # Scripts to define base LLM model, training hyper parameters, datasets and training codes
    │   ├── docker-entrypoint.sh           # run pipenv
    │   ├── docker-shell.sh                # Scripts to build the docker
    │   └── transform_new.py               # Scripts to reformat and split datasets
    ├── llm-rag
    │   ├── docker-volumes/chromadb/
    │   ├── input-datasets/books/
    │   ├── output_RAG_different_config/
    │   ├── outputs
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── cli.py                         # prepare necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB)
    │   ├── docker-compose.yml
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   └── llm-main.py
    ├── nutrition_predict_disease
    │   ├── models                         # Trained XGB
    │   ├── secrets
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   └── nutrition_predict_disease.py
    └── docker-shell.sh
```

## Milestone2 ###

In this milestone, we have the components for data management, including versioning, as well as the computer vision and language models.

#### Data:
We upload our datasets to the bucket, allowing the entire group to access them. Within the notebook folder, there is an EDA file that helps introduce the dataset information in more details. 

### Data Pipeline Containers (src/):
1. **food-classification:** The food-classification container recognizes the food in an image and stores the output back to Google Cloud Storage (GCS).
    - **Input:** Image and required secrets (provided via Docker)
    - **Output:** Name of the detected food name and predicted probability of the food
2. **food_to_nutrition:** The food_to_nutrition container links the food predicted from food_classification container & weight user inputs to the nutrient components and stores the output back to GCS.
    - **Input:** Food item predicted, the weight user inputs, and required secrets (provided via Docker)
    - **Output:** Nutrition components and calories of the food item identified
3. **nutrition_predict_disease:** The nutrition_predict_disease container use the nutritional components and calories of the food item from food_to_nutrition container to predict the risk of four chronic diseases. 
    - **Input:** Nutritional information of the food item, and required secrets (provided via Docker)
    - **Output:** Predicted probabilities for the risk of developing diseases, including obesity, diabetes, hypertension, and high cholesterol
4. llm-rag: This container is used to invoke fine-tuned model from fine-tuned LLM model with RAG to generate diatery suggestions
    - **Input:** Processed prompts containing previous outputs from container 2 and 3. The format of prompt is "Outputs from container 2 and 3, Could you give us some dietary advice based on these information?"
    - **Output:** The answer from fine-tuned LLM model with RAG to give response based on prompts.

### LLM Containers (src/):
1. **gemini-finetuner:** This container is used to process datasets used for fine-tuning and perform fine-tuning process in GCP.
    - **Input:** Processed Question Answering datasets as jsonl file, each entry has only question and answer parts.
    - **Output:** Fine-tuned LLM base model deployed as a endpoint for later RAG process.
  ```
  transform_new.py        # This process the format of nutrition question answering dataset and save this as jsonl files. Original dataset has instruction, input and output. After processing, question part includes instruction and inputs while answer part includes outputs.
  
  python cli.py --train   #  Fine-tune based model based on hyper parameters and datasets from bucket (all defined in src/gemini-finetuner/cli.py). Remember to deploy fine-tuned model as endpoint for later RAG usage.
  ```

2. **llm-rag:** Another container prepares data for the RAG model, including tasks such as chunking, embedding, and populating the vector database.
    - **Input:** Processed Raw data as txt. file, and user query text.
    - **Output:** Chunked data (.jsonl file), embedded data (.jsonl file), created ChromaDB instance, LLM response corresponding to the user query text, and LLM responses to our default evaluation questions (uploaded to GCP bucket as csv for different RAG configuration)
  ```
  python cli.py --chunk --chunk_type char-split
  python cli.py --chunk --chunk_type recursive-split
  ```


  - python cli.py --chunk --chunk_type char-split
  - python cli.py --chunk --chunk_type recursive-split
  This will read each text file in the input-datasets/books directory; Split the text into chunks using the specified method (character-based or recursive);Save the chunks as JSONL files in the outputs directory

  - python cli.py --embed --chunk_type char-split
  - python cli.py --embed --chunk_type recursive-split

  This will read the chunk files created in the previous section; Use Vertex AI's text embedding model to generate embeddings for each chunk; Save the chunks with their embeddings as new JSONL files.We use Vertex AI text-embedding-004 model to generate the embeddings

  - python cli.py --load --chunk_type char-split
  - python cli.py --load --chunk_type recursive-split

  This will connect to your ChromaDB instance, create a new collection (or clears an existing one), and load the embeddings and associated metadata into the collection.

  - python cli.py --chat --query_text={your specific input}

  This will generate the LLM response for specific user input using our vector database. Users could additionally specify “--chunk_type” to request two different vector base we generated using different split methods. 

  - python cli.py --process_questions --output-file={output file name}
  This will run our evaluation queries based on different RAG configuration and upload the results to the GCP buckets.

  #### src/RAG_on_fine_tuned_model/llm-main: 
   **Input:** Previous step results.
   **Output:** LLM response (uploaded to GCP bucket as .txt file)
  Running "python llm-main.py" could take the results of our previous step from the GCP bucket, which are the food nutrition info and the predicted diease risk, and then ask our fine-tuned RAG model for nutrition advice. The LLM response will automatically be uploaded to our GCP bucket.


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
   - src/llm-rag/outputs: Output of chunking and data embedding
   - src/llm-rag/output_RAG_different_contig: The log of LLM response with different RAG configuration
   - src/llm-rag/input-datasets/books: The preprocessed text of our raw data.
   - src/llm-rag/chromadb/: Data generated and used by Docker named chromadb, which is the container we used for setting up our vector database.

**Next Steps**
- Incorporate data versioning
- Combine some containers to reduce redundancy
  
