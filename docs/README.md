## Getting Started

The below commands will set up your virtual env and allow you to host this site on your localhost using port 8000. Make sure to run these commands from your root project dir
```sh
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r docs/requirements.txt
mkdocs serve
```
