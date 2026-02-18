#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    # 1. Download input artifact
    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # 2. Implementation of cleaning logic
    df = pd.read_csv(artifact_local_path)
    
    logger.info("Applying cleaning logic: filtering outliers and converting dates")
    
    # Use the dynamic parameters passed from MLflow/Argparse
    # idx = df['price'].between(args.min_price, args.max_price)
    idx = (df['price'] >= args.min_price) & (df['price'] <= args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # 3. Save the cleaned DataFrame
    # Using index=False is a requirement to pass Udacity's automated checks
    logger.info("Saving cleaned data to clean_sample.csv")
    df.to_csv("clean_sample.csv", index=False)

    # 4. Upload the artifact to W&B
    logger.info("Uploading artifact to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
    parser.add_argument("--input_artifact", type=str, help="Input artifact name", required=True)
    parser.add_argument("--output_artifact", type=str, help="Output artifact name", required=True)
    parser.add_argument("--output_type", type=str, help="Output artifact type", required=True)
    parser.add_argument("--output_description", type=str, help="Description for W&B", required=True)
    parser.add_argument("--min_price", type=float, help="Lower price bound", required=True)
    parser.add_argument("--max_price", type=float, help="Upper price bound", required=True)

    args = parser.parse_args()
    go(args)
