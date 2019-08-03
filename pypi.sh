#!/bin/sh

# 1. Update version number in setup.py
# 2. Update CHANGELOG
#
# For conda-forge
# 1. Fork the stumpy-feedstock
# 2. Create a new branch for the new version
# 3. In the recipe/meta.yaml file
#    a) Update version number on line 2
#    b) Update the sha256 on line 10 according to what is found on PyPI
#       in the "Download files" section of the left navigation pane
#    c) Reset the build number on line 14 since this is a new version
# 4. Commit the changes and push upstream for a PR
# 5. Check the checkboxes in the PR
# 6. Add a comment with "@conda-forge-admin, please rerender"
#
# To check that the distribution is valid, execute:
# twine check dist/* 

rm -rf dist
python3 setup.py sdist bdist_wheel 
twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*
#twine upload dist/*
