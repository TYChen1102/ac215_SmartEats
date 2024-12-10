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

### Project Organization
```
```

## Milestone5 

In this milestone, we deploy our application to a Kubernete cluster on GCP with manul scaling. We work on ansible playbooks to automate the provisioning and deployment of the infrastructure and application, including the kubernetes cluster. We set up a CI/CD pipeline using Github Actions, which runs unit test across every container, runs integration tests corss the explosed API on every pull request. We also implemented a ML workflow for 1 model. 

### Architectures:
#### Solution Architecture
![Solution Architecture](images/solution.jpg)
#### Technical Architecture
![Technical Architecture](images/tech.png)

### Containers


### Models:
1. **food-classification:** Recognizes the food in an image and stores the output back to Google Cloud Storage (GCS).
    - **Input:** Image and required secrets (provided via Docker)
    - **Output:** Name of the detected food name and predicted probability of the food
2. **food_to_nutrition:** Links the food predicted from food_classification container & weight user inputs to the nutrient components and stores the output back to GCS.
    - **Input:** Food item predicted, the weight user inputs, and required secrets (provided via Docker)
    - **Output:** Nutrition components and calories of the food item identified
3. **nutrition_predict_disease:** Uses the nutritional components and calories of the food item from food_to_nutrition container to predict the risk of four chronic diseases. 
    - **Input:** Nutritional information of the food item, and required secrets (provided via Docker)
    - **Output:** Predicted probabilities for the risk of developing diseases, including obesity, diabetes, hypertension, and high cholesterol
4. **llm-rag:** Invokes fine-tuned model from fine-tuned LLM model with RAG to generate diatery suggestions
    - **Input:** Processed prompts containing previous outputs from container 2 and 3. The format of prompt is "Outputs from container 2 and 3, Could you give us some dietary advice based on these information?"
    - **Output:** The answer from fine-tuned LLM model with RAG to give response based on prompts.
5. **gemini-finetuner:** Processes datasets used for fine-tuning and perform fine-tuning process in GCP.
    - **Input:** Processed Question Answering datasets as jsonl file, each entry has only question and answer parts.
    - **Output:** Fine-tuned LLM base model deployed as a endpoint for later RAG process.

2. **llm-rag:** Another container prepares data for the RAG model, including tasks such as chunking, embedding, and populating the vector database.
    - **Input:** Processed Raw data as txt. file, and user query text.
    - **Output:** Chunked data (.jsonl file), embedded data (.jsonl file), created ChromaDB instance, LLM response corresponding to the user query text, and LLM responses to our default evaluation questions (uploaded to GCP bucket as csv for different RAG configuration)

### Deploy on GCP:


### User Interfaces:


### Coverage Report:


