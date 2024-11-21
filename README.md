# AC215 - Milestone4 - SmartEats

### Project:
In this project, we aim to develop SmartEats, an AI-powered application that analyzes food images, predicts disease risks based on dietary habits, and offers personalized nutrition advice.
an application named SmartEats, which will take the food image as input from users and generate nutrition components with advise and potential disease risks. It will first require users to upload food pictures, and then it will use image recognition to estimate nutritional components like protein and fats and also display the amounts of each
component ideally. Based on the results, SmartEats will calculate calories, assess potential health risks, and provide tailored dietary suggestions, such as healthy recipes.

**Team Members:**
Jiayi Sun, Ninghui Hao, Qianwen Li, Taiyang Chen, Yantong Cui

**Group Name:**
The SmartEats Group

### Project Milestone 4 Organization

```
├── data
├── midterm_presentation
│   └── SmartEats_AC215_M3_slides.pdf      # midterm presentation slides
├── notebooks
│   ├── AC215 - EDA.pdf                    # EDA pdf file
│   ├── AC215_image_EfficientNet.ipynb     # train EfficientNet model
│   ├── AC215_image_VGG_new.ipynb          # train VGG model
│   ├── LLM-fintuning-Documentation.pdf    # Documentation of the LLM fine-tuning process
│   ├── LLM_RAG_preprocessing.ipynb        # construct the RAG vector database
│   ├── data_versioning_cloud_storage.ipynb# view version of the dataset
│   ├── dataset3_EDA&preprocessing.ipynb   # EDA for nutrtion-disease dataset
│   ├── frontpage_v2.html                  # HTML file for application front page
│   ├── frontpage_v2.jpg                   # screenshot of front page
│   ├── image_EDA.ipynb                    # EDA for image datasets
│   └── predict_disease_ML.ipynb           # fine-tuning of a XGBClassifier model
├── references
├── reports
│   ├── APCOMP215 Project Proposal.pdf
│   ├── DataPipeline1.jpg                  # Pipeline running screenshot
│   ├── DataPipeline2.jpg                  # Pipeline running screenshot
│   ├── Examples.pdf                       # Some example inputs&outputs
│   └── Final_output.png                   # Final output example
└── src
    ├── data-versioning
    │   ├── Dockerfile                     # To build the container for data versioning
    │   ├── Pipfile                        # Define packages used in data versioning
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   └── smarteats_data.dvc
    ├── food-classification
    │   ├── secrets
    │   ├── Dockerfile                     # To build the container for food classification
    │   ├── Pipfile                        # Define packages used in food classification
    │   ├── Pipfile.lock
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
    │   ├── Pipfile.lock
    │   ├── cli.py                         # Scripts to define base LLM model, training hyper parameters, datasets and training codes
    │   ├── docker-entrypoint.sh           # run pipenv
    │   ├── docker-shell.sh                # Scripts to build the docker
    │   └── transform_new.py               # Scripts to reformat and split datasets
    ├── llm-rag
    │   ├── docker-volumes/chromadb/       # Data generated and used by Docker named chromadb, which is the container we used for setting up our vector database.
    │   ├── input-datasets/books/          # The preprocessed text of our raw data.
    │   ├── output_RAG_different_config/   # The log of LLM response with different RAG configuration
    │   ├── outputs                        # Output of chunking and data embedding
    │   ├── Dockerfile                     # Build container for LLM RAG
    │   ├── Pipfile                        # Define packages
    │   ├── Pipfile.lock
    │   ├── cli.py                         # prepare necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB)
    │   ├── docker-compose.yml
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   └── llm-main.py                    # invoke fine-tuned model from fine-tuned LLM model with RAG to generate diatery suggestions
    ├── nutrition_predict_disease
    │   ├── models                         # Trained XGBClassifier models
    │   ├── secrets
    │   ├── Dockerfile                     # To build container to predict disease risks based on nutrition content
    │   ├── Pipfile                        # Define necessary packages and requirements
    │   ├── Pipfile.lock
    │   └── nutrition_predict_disease.py   # Load models and dataset to predict diseases risks
    ├── secrets
    └── docker-shell.sh                    # Combine 4 containers in Data Pipeline (see below) and print final suggestions from fine-tuned LLm with RAG

```

## Running Backend and Frontend
```
cd src_1container    # move into the directory with docker-shell.sh
sh docker-shell.sh   # Then, the backend is activated
# now while runing the backend container, also run a frontend container
cd frontend          # move into the frontend directory with docker-shell.sh
sh dokcer-shell.sh   # Then, the frontend is activated
http-server          # activate the server
```
Visit http://localhost:8080/model.html

## Application Design

### Application architecture
#### Solution Architecture
#### Technical Architecture


### User interface
<img width="1512" alt="image" src="https://github.com/user-attachments/assets/77e7f74c-f6d0-45eb-a85a-38e64657578f">

## Continuous Integration Setup:
We have built two functioning CI pipelines that run on every push or merge.
- Pre-commit checks: Automated build process and code quality checks using linting tools Flake8 running on GitHub Actions.
- Continuous Integration and Continuous Deployment: Execution of unit, integration and systems tests with test results reported.

## Test Documentation:
Detailed test case descriptions and test documentation is located in the directory **reports/Test_Documentation.pdf**

#### Testing Tool Used
- PyTest: Used as the primary testing framework.
- unittest: Provides tools like patch and MagicMock to mock external dependencies such as APIs and models

#### Testing Strategy
- Unit Tests
  - Validates individual functions such as process_image, food_to_nutrition, and transform_data.
  - Ensures correctness of outputs for given inputs.
  - External APIs (e.g., USDA API) and machine learning models are mocked to simulate their behavior during tests.
- Integration Tests
  - Verifies that multiple components work together as expected, such as preprocessing an image and running it through the classification pipeline.
- System Tests
  - Covers user flows and interactions, including building the container and running all steps in the app.
