# Pull Request Checklist

Below is a simple checklist but please do not hesitate to ask for assistance!

- [ ] Fork, clone, and checkout the newest version of the code
- [ ] Create a new branch
- [ ] Make necessary code changes
- [ ] Install `black` (i.e., `python -m pip install black` or `conda install -c conda-forge black`)
- [ ] Install `flake8` (i.e., `python -m pip install flake8` or `conda install -c conda-forge flake8`)
- [ ] Install `pytest-cov` (i.e., `python -m pip install pytest-cov` or `conda install -c conda-forge pytest-cov`)
- [ ] Run `black --exclude=".*\.ipynb" --extend-exclude=".venv" --diff ./` in the root stumpy directory
- [ ] Run `flake8 --extend-exclude=.venv ./` in the root stumpy directory
- [ ] Run `./setup.sh dev && ./test.sh` in the root stumpy directory
- [ ] Reference a Github issue (and create one if one doesn't already exist)
