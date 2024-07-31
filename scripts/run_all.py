import os
import subprocess

# TMP: Script to run all pipelines on mega-ni-data docs
# In the future, will implement utils so that the output directories are automatically set
# based on hash of input data and pipeline configuration

# The script implements a basic CLI to run the pipelines on the mega-ni-data docs
# Inputs are the path to the mega-ni-data docs
# Default output directtory is ../ouputs/
# The script will create a subdirectory for the hash on the input data
# Then execute the {pipeline_name}/run.py script with the input data and save the output to the output directory
# The pipeline is responsible for saving the output in the correct format

def run_pipeline(pipeline_name, input_data_path, output_dir):
    # Create a subdirectory for the hash of the input data
    hash_dir = os.path.join(output_dir, hash(input_data_path))
    os.makedirs(hash_dir, exist_ok=True)
    
    # Execute the {pipeline_name}/run.py script with the input data
    pipeline_script = os.path.join(pipeline_name, "run.py")
    subprocess.run(["python", pipeline_script, input_data_path, hash_dir])

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_path", help="Path to the input data")
    parser.add_argument("--output_dir", default="../outputs/", help="Path to the output directory")
    parser.add_argument("--pipelines", nargs="+", help="List of pipelines to run")

    args = parser.parse_args()

    # Run the pipelines
    for pipeline_name in args.pipelines:
        run_pipeline(pipeline_name, args.input_data_path, args.output_dir)

if __name__ == "__main__":
    main()