### Python environment set up
1. Install uv for python and dependencies management

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install python versions if needed

```bash
# Install python if not already installed
uv python install 3.10 3.11 3.12
```

3. Create a 3.11 python virtual environment with uv

```bash
uv venv --python=3.11
```

4. Activate the python virtual environment

```bash
source .venv/bin/activate
```

5. Install python dependencies with uv

```bash
uv pip install -r requeriments.txt
```

### System Dependencies Installation
6. Install FFmpeg (It is a common tool for all ultiagent systems to manage audio files)

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install ffmpeg

```
#### Web Scraping & Search Setup

7. Install and setup crawl4ai

```bash
uv pip install -U crawl4ai
```

8. Setup crawl4ai
```bash
# Setup crawl4ai
crawl4ai-setup
crawl4ai-doctor
```

9. Install playwright browsers

```bash
playwright install
```

10. Create the gaia run directory

```bash
mkdir gaia_run
```

11. Execute a single task

```bash
uv run evaluate_agent.py --task_id 4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2
```

12. Execute a block of task

```bash
uv run evaluate_agent.py --start 0 --end 2
```



