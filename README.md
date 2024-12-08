# AC215 - SmartEats 
## This is the project for APCOMP215, 2024 Fall. 

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
```

## Milestone5 

In this milestone, we deploy our application to a Kubernete cluster on GCP with manul scaling. We work on ansible playbooks to automate the provisioning and deployment of the infrastructure and application, including the kubernetes cluster. We set up a CI/CD pipeline using Github Actions, which runs unit test across every container, runs integration tests corss the explosed API on every pull request. We also implemented a ML workflow for 1 model. 


### Containers


### Models:
1. **food-classification:** The food-classification container recognizes the food in an image and stores the output back to Google Cloud Storage (GCS).
    - **Input:** Image and required secrets (provided via Docker)
    - **Output:** Name of the detected food name and predicted probability of the food
2. **food_to_nutrition:** The food_to_nutrition container links the food predicted from food_classification container & weight user inputs to the nutrient components and stores the output back to GCS.
    - **Input:** Food item predicted, the weight user inputs, and required secrets (provided via Docker)
    - **Output:** Nutrition components and calories of the food item identified
3. **nutrition_predict_disease:** The nutrition_predict_disease container use the nutritional components and calories of the food item from food_to_nutrition container to predict the risk of four chronic diseases. 
    - **Input:** Nutritional information of the food item, and required secrets (provided via Docker)
    - **Output:** Predicted probabilities for the risk of developing diseases, including obesity, diabetes, hypertension, and high cholesterol
4. **llm-rag:** This container is used to invoke fine-tuned model from fine-tuned LLM model with RAG to generate diatery suggestions
    - **Input:** Processed prompts containing previous outputs from container 2 and 3. The format of prompt is "Outputs from container 2 and 3, Could you give us some dietary advice based on these information?"
    - **Output:** The answer from fine-tuned LLM model with RAG to give response based on prompts.

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
  # Read each text file in the input-datasets/books directory
  # Split the text into chunks using the specified method (character-based or recursive)
  # Save the chunks as JSONL files in the outputs directory
  python cli.py --chunk --chunk_type char-split
  python cli.py --chunk --chunk_type recursive-split
  ```

  ```
  # Read the chunk files created in the previous section;
  # Use Vertex AI's text embedding model to generate embeddings for each chunk;
  # Save the chunks with their embeddings as new JSONL files.We use Vertex AI text-embedding-004 model to generate the embeddings
  python cli.py --embed --chunk_type char-split
  python cli.py --embed --chunk_type recursive-split
  ```

  ```
  # Connect to your ChromaDB instance
  # Create a new collection (or clears an existing one)
  # load the embeddings and associated metadata into the collection.
  python cli.py --load --chunk_type char-split
  python cli.py --load --chunk_type recursive-split
  ```

  ```
  # Generate the LLM response for specific user input using our vector database. Users could additionally specify “--chunk_type” to request two different vector base we generated using different split methods. 
  python cli.py --chat --query_text={your specific input}
  ```

  ```
  # This will run our evaluation queries based on different RAG configuration and upload the results to the GCP buckets.
  python cli.py --process_questions --output-file={output file name}
  ```
