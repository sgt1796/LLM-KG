import subprocess

def run_command(command):
    """Run a shell command and print its output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        exit(result.returncode)

# Step 1: Fetch Dataset1 Papers
run_command("python fetch_ncbi.py --csv dataset1-AlzheimerD-set.csv --out papers_dataset1 --resume")

# Step 2: Sample 50 Papers from Dataset1 SKIPPED
#run_command("bash sample_papers.sh 50 papers_dataset1 sampled_papers_dataset1")

# Step 3: Run KG Extraction for Dataset1 Papers with LLMNER
run_command("python main.py --pdf papers_dataset1 --output graph_dataset1_llm.json --ner ollama --chunking abstract_discussion")

# Step 4: Deduplicate and Clean the Dataset1 Graph
#run_command("python dedupe.py --input graph_dataset1_llm.json --output graph_clean_dataset1_llm.json --spacy-model en_core_sci_sm --min-support 3 --verbose-every 50000")

# Step 5: Visualize the Cleaned Dataset1 Graph
run_command("python KG_visualizer.py --k-core 1 --max-nodes 300 --label-top 10 --highlight-edges-top 20 --edge-labels-top 10 --min-weight 5 --input graph_dataset1_llm.json --layout spring --largest-only")

# Step 6: Visualize using Pyvis
run_command("python pyvis_view.py --input graph_dataset1_llm.json --html graph_dataset1_llm.html --weight \">=0\" --k-core 0 --max-nodes 500 --max-edges 600 --label-top 20 --physics barnesHut")