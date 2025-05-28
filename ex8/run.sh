#!/bin/bash

source ~/.zshrc

# cd /home/username/Projects/ProjectName

# If you use Poetry
poetry env use /home/{username}/{workspace名}/vanilla_VAE/.venv/bin/python
poetry env info
poetry install

poetry run python main.py main --z_dim 2