import os
import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb
from google.cloud import storage
# Vertex AI
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig

# Langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
MODEL_ENDPOINT = "projects/1058117673285/locations/us-central1/endpoints/6472676518647562240" # our fine-tuned model
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000
bucket_name = "ac215smarteat"
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#python
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 8192,  # Maximum number of tokens for output
    "temperature": 0.25,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}
# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in nutrition and diet knowledge. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response using only the information found in the given chunks.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a nutrition and diet expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.

Remember:
- You are an expert in nutrition and diet, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- If asked about topics unrelated to cheese, politely redirect the conversation back to nutrition-and-diet-related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.

Your goal is to provide accurate, helpful information about nutrition and diet based solely on the content of the text chunks you receive with each query.
"""
generative_model = GenerativeModel(
	MODEL_ENDPOINT,
	system_instruction=[SYSTEM_INSTRUCTION]
)

book_mappings = {
	"Dietary_Guidelines_for_Americans_2020-2025.txt": {"author":"Dietary Guidelines Writing Team", "year": 2020-2025},
	"wiki_book.txt": {"author":"wiki writers", "year": 2023}

}

rag_settings = {
    	"Embedding Model": EMBEDDING_MODEL ,
    	"generation_config": generation_config,
		"retrieval_parameters": {
        	"Number_of_top_documents_fetched": 10,
    	},
		"chunking method": "char-split"
    }

def generate_query_embedding(query):
	query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
	kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
	embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
	return embeddings[0].values


def generate_text_embeddings(chunks, dimensionality: int = 256, batch_size=250):
	# Max batch size is 250 for Vertex AI
	all_embeddings = []
	for i in range(0, len(chunks), batch_size):
		batch = chunks[i:i+batch_size]
		inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in batch]
		kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
		embeddings = embedding_model.get_embeddings(inputs, **kwargs)
		all_embeddings.extend([embedding.values for embedding in embeddings])

	return all_embeddings


def load_text_embeddings(df, collection, batch_size=500):

	# Generate ids
	df["id"] = df.index.astype(str)
	hashed_books = df["book"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
	df["id"] = hashed_books + "-" + df["id"]

	metadata = {
		"book": df["book"].tolist()[0]
	}
	if metadata["book"] in book_mappings:
		book_mapping = book_mappings[metadata["book"]]
		metadata["author"] = book_mapping["author"]
		metadata["year"] = book_mapping["year"]

	# Process data in batches
	total_inserted = 0
	for i in range(0, df.shape[0], batch_size):
		# Create a copy of the batch and reset the index
		batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

		ids = batch["id"].tolist()
		documents = batch["chunk"].tolist()
		metadatas = [metadata for item in batch["book"].tolist()]
		embeddings = batch["embedding"].tolist()

		collection.add(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings
		)
		total_inserted += len(batch)
		print(f"Inserted {total_inserted} items...")

	print(f"Finished inserting {total_inserted} items into collection '{collection.name}'")


def chunk(method="char-split"):
	print("chunk()")

	# Make dataset folders
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)

	# Get the list of text file
	text_files = glob.glob(os.path.join(INPUT_FOLDER, "books", "*.txt"))
	print("Number of files to process:", len(text_files))

	# Process
	for text_file in text_files:
		print("Processing file:", text_file)
		filename = os.path.basename(text_file)
		book_name = filename.split(".")[0]

		with open(text_file) as f:
			input_text = f.read()

		text_chunks = None
		if method == "char-split":
			chunk_size = 350
			chunk_overlap = 20
			# Init the splitter
			text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)

			# Perform the splitting
			text_chunks = text_splitter.create_documents([input_text])
			text_chunks = [doc.page_content for doc in text_chunks]
			print("Number of chunks:", len(text_chunks))

		elif method == "recursive-split":
			chunk_size = 350
			# Init the splitter
			text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size)

			# Perform the splitting
			text_chunks = text_splitter.create_documents([input_text])
			text_chunks = [doc.page_content for doc in text_chunks]
			print("Number of chunks:", len(text_chunks))

		if text_chunks is not None:
			# Save the chunks
			data_df = pd.DataFrame(text_chunks,columns=["chunk"])
			data_df["book"] = book_name
			print("Shape:", data_df.shape)
			print(data_df.head())

			jsonl_filename = os.path.join(OUTPUT_FOLDER, f"chunks-{method}-{book_name}.jsonl")
			with open(jsonl_filename, "w") as json_file:
				json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="char-split"):
	print("embed()")

	# Get the list of chunk files
	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	# Process
	for jsonl_file in jsonl_files:
		print("Processing file:", jsonl_file)

		data_df = pd.read_json(jsonl_file, lines=True)
		print("Shape:", data_df.shape)
		print(data_df.head())

		chunks = data_df["chunk"].values
		embeddings = generate_text_embeddings(chunks,EMBEDDING_DIMENSION, batch_size=100)
		data_df["embedding"] = embeddings

		# Save
		print("Shape:", data_df.shape)
		print(data_df.head())

		jsonl_filename = jsonl_file.replace("chunks-","embeddings-")
		with open(jsonl_filename, "w") as json_file:
			json_file.write(data_df.to_json(orient='records', lines=True))


def load(method="char-split"):
	print("load()")

	# Connect to chroma DB
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"
	print("Creating collection:", collection_name)

	try:
		# Clear out any existing items in the collection
		client.delete_collection(name=collection_name)
		print(f"Deleted existing collection '{collection_name}'")
	except Exception:
		print(f"Collection '{collection_name}' did not exist. Creating new.")

	collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
	print(f"Created new empty collection '{collection_name}'")
	print("Collection:", collection)

	# Get the list of embedding files
	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	# Process
	for jsonl_file in jsonl_files:
		print("Processing file:", jsonl_file)

		data_df = pd.read_json(jsonl_file, lines=True)
		print("Shape:", data_df.shape)
		print(data_df.head())

		# Load data
		load_text_embeddings(data_df, collection)


def chat(method="char-split", query_text=None):  # change it to allows chat query
    print("chat()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    if query_text is None:
        query_text = "What is protein food?"
    query_embedding = generate_query_embedding(query_text)

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Query based on embedding value
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    # Prepare the document content to avoid backslash issue
    documents_text = "\n".join(results["documents"][0])
    # Construct the input prompt
    INPUT_PROMPT = f"""
    {query_text}
    {documents_text}
    """

    # Generate response
    response = generative_model.generate_content(
        [INPUT_PROMPT],  # Input prompt
        generation_config=generation_config,  # Configuration settings
        stream=False,  # Enable streaming for responses, originally False
    )
    generated_text = response.text
    print("LLM Response:", generated_text)

    # Upload output to GCP bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    output_file_name = 'final_step_LLM_output.txt'

    # Open the text file in write mode and save the generated text
    with open(output_file_name, 'w', encoding='utf-8') as file:
        file.write(generated_text)

    blob = bucket.blob(f"shared_results/{output_file_name}")
    blob.upload_from_filename(output_file_name)
    print('Output uploaded to GCP bucket.')

    return generated_text, results["documents"]


def evaluate_and_save_to_csv(questions, output_file, method="char-split"):

    # Loop through the list of questions, generate an LLM response for each, and record both
	results_list = []

	for question in questions:
		print(f"Processing question: {question}")
		response,docs = chat(method=method, query_text=question)
		results_list.append({
            "RAG_config": rag_settings,
            "question": question,
            "llm_response": response
        })
		time.sleep(60) # do not overly request quota
	results_df = pd.DataFrame(results_list)



	results_df.to_csv(output_file, index=False)
     # Upload output to GCP bucket
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(f'LLM_evaluation/{output_file}')
	blob.upload_from_filename(output_file)
	print('LLM evaluation output uploaded to GCP bucket.')


def main(args=None):
	# The ten evaluation questions
	questions = [
        "What are the key nutritional considerations for school-aged children according to the Dietary Guidelines, and how do school meal programs contribute to their dietary needs?",
        "How do the Dietary Guidelines address the unique nutritional needs of infants and toddlers, and what are the key recommendations for this age group?",
        "Describe the recommended Healthy Dietary Patterns for adults aged 19-59, as outlined in the Dietary Guidelines, and the rationale behind these recommendations.",
        "Discuss the role of plant proteins in human nutrition and their significance as highlighted in the text 'Understanding Nutrition.'",
        "What findings does the chapter on older adults provide regarding nutrient needs and dietary patterns, and how do these change with age?",
        "How do the Dietary Guidelines suggest supporting healthy dietary patterns among pregnant or lactating women, and what are some specific nutritional challenges they face?",
        "Identify and explain the implications of underconsumption of nutrients of public health concern throughout different life stages as described in the Dietary Guidelines.",
        "How to manage dietary habits to mitigate the risk of chronic health conditions?",
        "Describe the importance of recording supplementation dosages in nutritional assessments and its implications for disease prevention and management.",
        "How is body mass index (BMI) used as a risk factor for diseases in the context of nutritional assessments, and what are its limitations?"
    ]
	print("CLI Arguments:", args)

	if args.process_questions:evaluate_and_save_to_csv(questions=questions, output_file=args.output_file, method=args.chunk_type)
	if args.chunk:
		chunk(method=args.chunk_type)

	if args.embed:
		embed(method=args.chunk_type)

	if args.load:
		load(method=args.chunk_type)

	if args.chat:
		chat(method=args.chunk_type,query_text=args.query_text)


	if args.process_questions:
		evaluate_and_save_to_csv(questions=questions, output_file=args.output_file, method=args.chunk_type)



if __name__ == "__main__":
	# Generate the inputs arguments parser
	# if you type into the terminal '--help', it will provide the description
	parser = argparse.ArgumentParser(description="CLI")

	parser.add_argument(
		"--chunk",
		action="store_true",
		help="Chunk text",
	)
	parser.add_argument(
		"--embed",
		action="store_true",
		help="Generate embeddings",
	)
	parser.add_argument(
		"--load",
		action="store_true",
		help="Load embeddings to vector db",
	)
	parser.add_argument(
		"--chat",
		action="store_true",
		help="Chat with LLM",
	)
	parser.add_argument("--chunk_type", default="char-split", help="char-split | recursive-split")
	parser.add_argument(
    "--query_text",
    type=str,
    help="Query text for vector database and for chat",
	)
	parser.add_argument("--process_questions", action="store_true", help="Process a list of predefined questions using RAG and output responses to CSV")
	parser.add_argument("--output-file", type=str, help="Output CSV file for questions, retrieved contexts, and responses")

	args = parser.parse_args()

	main(args)
