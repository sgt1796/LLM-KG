import subprocess

def run_command(command):
    """Run a shell command and print its output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        exit(result.returncode)

# Step 1: Fetch ADHD Papers
#run_command("python fetch_ncbi.py --csv csv-adhd-set.csv --out papers_ADHD")

# Step 2: Fetch SUD Papers
#run_command("python fetch_ncbi.py --csv csv-substanceu-set.csv --out papers_SUD")

# Step 3: Sample 10 Papers from ADHD
run_command("bash sample_papers.sh 10 papers_ADHD sampled_papers_adhd")

# Step 4: Sample 10 Papers from SUD
run_command("bash sample_papers.sh 100 papers_SUD sampled_papers_sud")

# Step 5: Run KG Extraction for ADHD Papers with LLMNER
run_command("python main.py --pdf sampled_papers_adhd --output graph_adhd_llm.json --ner llm")

# Step 6: Run KG Extraction for SUD Papers with LLMNER
run_command("python main.py --pdf sampled_papers_sud --output graph_sud_llm.json --ner llm")

# Step 7: Merge ADHD and SUD Graphs
run_command("python main.py --pdf sampled_papers_sud --output graph_combined_llm.json --merge graph_adhd_llm.json --ner llm")

# Step 8: Deduplicate and Clean the Combined Graph
run_command("python dedupe.py --input graph_combined_llm.json --output graph_clean_llm.json --spacy-model en_core_sci_sm --min-support 3 --verbose-every 50000")

# Step 9: Visualize the Cleaned Graph
run_command("python KG_visualizer.py --k-core 3 --max-nodes 300 --label-top 10 --highlight-edges-top 20 --edge-labels-top 10 --min-weight 50 --input graph_clean_llm.json --layout spring --largest-only")