"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip



GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
GCP_REGION = os.environ["GCP_REGION"]

# DATA_COLLECTOR_IMAGE = "gcr.io/ac215-project/cheese-app-data-collector"
DATA_COLLECTOR_IMAGE = "nih684/smarteat-data-collector"
DATA_PROCESSOR_IMAGE = "nih684/smarteat-data-processor"
MODEL_PREDICTION_IMAGE = "nih684/smarteat-model-prediction"

#DATA_COLLECTOR_IMAGE = "dlops/cheese-app-data-collector"
#DATA_PROCESSOR_IMAGE = "dlops/cheese-app-data-processor"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def data_collector():
    print("data_collector()")

    # Define a Container Component
    @dsl.container_component
    def data_collector():
        container_spec = dsl.ContainerSpec(
            image=DATA_COLLECTOR_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--search",
                "--nums 10",
                "--query pizza",
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec

    # Define a Pipeline
    @dsl.pipeline
    def data_collector_pipeline():
        data_collector()

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        data_collector_pipeline, package_path="data_collector.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "smarteat-data-collector-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="data_collector.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    #job.run(service_account=GCS_SERVICE_ACCOUNT)
    job.run(service_account="ml-workflow@ac215-smarteat-437821.iam.gserviceaccount.com")



def data_processor():
    print("data_processor()")

    # Define a Container Component for data processor
    @dsl.container_component
    def data_processor():
        container_spec = dsl.ContainerSpec(
            image=DATA_PROCESSOR_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--clean",
                #"--prepare",
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec

    # Define a Pipeline
    @dsl.pipeline
    def data_processor_pipeline():
        data_processor()

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        data_processor_pipeline, package_path="data_processor.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "smarteat-data-processor-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="data_processor.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    #job.run(service_account=GCS_SERVICE_ACCOUNT)
    job.run(service_account="ml-workflow@ac215-smarteat-437821.iam.gserviceaccount.com")


def model_predicting():
    print("model_predicting()")

    # Define a Container Component for model predicting
    @dsl.container_component
    def model_predicting():
        container_spec = dsl.ContainerSpec(
            image=MODEL_PREDICTION_IMAGE,
            command=[],
            args=[
                "new_predict_food.py"
            ],
        )
        return container_spec
    
    # Define a Pipeline
    @dsl.pipeline
    def model_predicting_pipeline():
        model_predicting()

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        model_predicting_pipeline, package_path="model_predicting.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "smarteat-model-predicting-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="model_predicting.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    #job.run(service_account=GCS_SERVICE_ACCOUNT)
    job.run(service_account="ml-workflow@ac215-smarteat-437821.iam.gserviceaccount.com")



def pipeline():
    print("pipeline()")
    # Define a Container Component for data collector
    @dsl.container_component
    def data_collector():
        container_spec = dsl.ContainerSpec(
            image=DATA_COLLECTOR_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--search",
                "--nums 10",
                "--query pizza",
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec

    # Define a Container Component for data processor
    @dsl.container_component
    def data_processor():
        container_spec = dsl.ContainerSpec(
            image=DATA_PROCESSOR_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--clean",
                
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec
    
    @dsl.container_component
    def model_predicting():
        container_spec = dsl.ContainerSpec(
            image=MODEL_PREDICTION_IMAGE,
            command=[],
            args=[
                "new_predict_food.py"
            ],
        )
        return container_spec


    # Define a Pipeline
    @dsl.pipeline
    def ml_pipeline():
        # Data Collector
        data_collector_task = (
            data_collector()
            .set_display_name("Data Collector")
            .set_cpu_limit("500m")
            .set_memory_limit("2G")
        )
        # Data Processor
        data_processor_task = (
            data_processor()
            .set_display_name("Data Processor")
            .after(data_collector_task)
        )
        # Model Predicting
        model_predicting_task = (
            model_predicting()
            .set_display_name("Model Predicting")
            .after(data_processor_task)
        )
        

    # Build yaml file for pipeline
    compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "smarteat-pipeline-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    #job.run(service_account=GCS_SERVICE_ACCOUNT)
    job.run(service_account="ml-workflow@ac215-smarteat-437821.iam.gserviceaccount.com")



def sample_pipeline():
    print("sample_pipeline()")
    # Define Component
    @dsl.component
    def square(x: float) -> float:
        return x**2

    # Define Component
    @dsl.component
    def add(x: float, y: float) -> float:
        return x + y

    # Define Component
    @dsl.component
    def square_root(x: float) -> float:
        return x**0.5

    # Define a Pipeline
    @dsl.pipeline
    def sample_pipeline(a: float = 3.0, b: float = 4.0) -> float:
        a_sq_task = square(x=a)
        b_sq_task = square(x=b)
        sum_task = add(x=a_sq_task.output, y=b_sq_task.output)
        return square_root(x=sum_task.output).output

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        sample_pipeline, package_path="sample-pipeline1.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "sample-pipeline-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="sample-pipeline1.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    #job.run(service_account=GCS_SERVICE_ACCOUNT)
    job.run(service_account="ml-workflow@ac215-smarteat-437821.iam.gserviceaccount.com")



def main(args=None):
    print("CLI Arguments:", args)

    if args.data_collector:
        data_collector()

    if args.data_processor:
        print("Data Processor")
        data_processor()

    if args.model_predicting:
        print("Model Predicting")
        model_predicting()

    if args.pipeline:
        pipeline()

    if args.sample:
        print("Sample Pipeline")
        sample_pipeline()

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "--data_collector",
        action="store_true",
        help="Run just the Data Collector",
    )
    parser.add_argument(
        "--data_processor",
        action="store_true",
        help="Run just the Data Processor",
    )
    parser.add_argument(
        "--model_predicting",
        action="store_true",
        help="Run just Model Predicting",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="SmartEats Pipeline",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample Pipeline 1",
    )

    args = parser.parse_args()

    main(args)
