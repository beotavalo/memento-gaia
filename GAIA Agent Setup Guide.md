## Manual tasks

1. Clone the repository

```bash
# Clone repository
git clone https://github.com/beotavalo/memento-gaia.git
cd memento-gaia
```


### Prepare GAIA dataset
2. Install Git LFS. 
```bash
git lfs install
```
> **⚠️ Important**: 
> It is important to clone correctly the dataset files 
> You only need to run this command once per machine:
3. (Manual Task) Get Your Hugging Face API Token

> **⚠️ Important**: 
> Go to the Hugging Face website and log in.
> Click on your profile picture in the top-right corner and go to Settings.
> In the left sidebar, navigate to Access Tokens.
> Click the New token button. Give it a name (e.g., "My Laptop") and assign it a write role, which is necessary for pushing/uploading.
> Copy the generated token. Be careful, as it will only be shown once.

4. Install Hugging Face library.

```bash
uv pip install -U huggingface_hub
```

5. Run the HF login command: 

```bash
git config --global credential.helper store
hf auth login
```

6. Download the GAIA dataset from Hugging Face


```bash
mkdir dataset && cd dataset
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA
```

7. Create the gaia run directory

```bash
mkdir gaia_run
```

### Environment Variables Configuration
8. Create an `.env` file to access `API_KEYS`

After creating the `.env` file, you need to configure the following API keys and service endpoints:

```bash
# OPENAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # or your custom endpoint

#===========================================
# Tools & Services API
#===========================================
# Chunkr API (https://chunkr.ai/)
CHUNKR_API_KEY=your_chunkr_api_key_here

# Jina API
JINA_API_KEY=your_jina_api_key_here

# ASSEMBLYAI API 
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

**Note**: Replace `your_*_api_key_here` with your actual API keys. Some services are optional depending on which tools you plan to use.

### SearxNG Setup
9. Set up SearxNG
For web search capabilities, set up SearxNG: 
You can follow https://github.com/searxng/searxng-docker/ to set the docker and use our setting.

```bash
# In a new terminal
cd ./Memento/searxng-docker
docker compose up -d
```
## Automated Python and System Set Up
This is the step by step guide for python environment and system set up 
[Step by step set up](https://github.com/beotavalo/memento-gaia/blob/main/gaia-agent-plan/plan-step-by-step.md)

### Code agent promt

We suggest to execute this Prompt with code agents (Cursor, gemini-cli, claude code)

```
Run step by step the @gaia-agent-plan/plan-step-by-step.md, log the issues in @gaia-agent-plan/set_up_log.md, and register the progress in @gaia-agent-plan/set_up_checklist.md
```

### Basic Usage
Execute a single task

```bash
uv run evaluate_agent.py --task_id 4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2
```

Execute a block of task

```bash
uv run evaluate_agent.py --start 0 --end 2
```

## Configuration

### Model Selection

- **Planner Model**: Defaults to `gpt-5` for task decomposition
- **Executor Model**: Defaults to `gpt-5` for task execution
- **Custom Models**: Support for any OpenAI-compatible API

### Dataset Path Configuration

To evaluate test dataset set the dataset path on line 345 of `evaluate_agent.py`
```bash
validation_data_path = os.path.join(os.path.dirname(__file__), 'dataset/GAIA/2023/test/metadata.jsonl')
```

### Tool Configuration

- **Search**: Configure SearxNG instance URL
- **Code Execution**: Customize import whitelist and security settings
- **Document Processing**: Set cache directories and processing limits

---
