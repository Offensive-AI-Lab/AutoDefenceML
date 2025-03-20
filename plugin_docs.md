

## Adding a New Attack

Follow these steps to add a new attack to the system using the `art` library.

1. **Prepare the Attack Class in `art` Format**
   - Ensure your attack class follows the `art` library structure and includes a `generate` method.

2. **Upload Your Attack as a Python Package to Artifact Registry - Can skip this step if you've installed the art_plugin**

   1. **Navigate to Artifact Registry**
      - Open the GCP console and search for "Artifact Registry."

   2. **Create a Repository**
      - Click on “Create repository” at the top of the page.
      - Choose the repository type (in this case, select **Python**).
      - Complete the creation process to have a repository ready for saving your package.

   3. **Upload the Python Package**
      - **Install necessary tools**:
        ```bash
        pip install twine keyrings.google-artifactregistry-auth wheel
        ```
      - **Specify package version**:
        Ensure you have a `setup.py` file that includes the following structure:
        ```python
        from setuptools import setup, find_packages

        setup(
           name="art_attacks_plugin",
           version="0.2",
           author="MABADATA",
           author_email="mabadatabgu@gmail.com",
           description="New ART attacks plugins",
           include_package_data=True,
           classifiers=[
               "Programming Language :: Python :: 3",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
           ],
           dependency_links=[
               'https://pypi.python.org/simple'
           ],
           package_dir={"": "src"},
           packages=find_packages(where="src"),
           python_requires=">=3.6",
        )
        ```
      - **Build the package**:
        ```bash
        python setup.py bdist_wheel
        ```
      - **Upload to Artifact Registry**:
        ```bash
        twine upload --repository-url <repo-url>/<repo-name> dist/*
        ```

3. **Install the Package from Artifact Registry in the Backend Project**
   - In the `Dockerfile` of the backend project, add the following line to install the package:
     ```Dockerfile
     RUN pip install --index-url <repo-url> <package-name>
     ```
   - Example:
     ```Dockerfile
     RUN pip install --index-url https://me-west1-python.pkg.dev/autodefenseml/art-attacks-plugin/simple/ art-attacks-plugin
     ```

4. **Add the Attack to the `art-handler` Repository**

   - Clone the `art-handler` repository from GitHub if you haven’t already.
   - In the `attack_defense_map.json` file, add the new attack details. For example:
     ```json
     "29": {
       "name": "Boundary_Tabular",
       "class_name": "art_attacks_plugin.Boundary_Tabular.BoundaryAttack",
       "type": "Evasion",
       "violation": "Integrity",
       "assumption": "BB",
       "influence": "Exploratory",
       "p-norm": "inf",
       "run_time": "Long",
       "description": "Implementation of the boundary attack tabular version by Fraidi Yael Itzhakev | Paper link: https://arxiv.org/abs/1712.04248",
       "default_max_iter": 5000
     }
     ```
   - Open the `setup.py` file in the `art-handler` repository, increase the version number, and rebuild the package:
     ```bash
     python setup.py bdist_wheel
     ```
   - Upload the package to Artifact Registry:
     ```bash
     twine upload --repository-url https://us-central1-python.pkg.dev/autodefenseml/art-handler/ dist/*
     ```

5. **Import the Attack Package in Code**
   - In the appropriate file (e.g., `attack_defense_validation.py`), import your attack package:
     ```python
     from <package_name> import *
     ```

6. **Make Additional Changes as Needed**
   - If your attack requires additional parameters or configurations, or if you want to bable to optimize the parameters during evaluation, make the necessary addition modifications in `attack_defense_validation.py` (in the `pre_run` package) and `model_evaluation.py` (in the `eval` package). For example, add the new attack to the hyperparameter optimization list [here](https://github.com/Offensive-AI-Lab/AutoDefenceML/blob/a8a3dfe07419b5853ba6b160054ba00c3dbfefeb/Backend/api/src/eval/evaluation/model_evaluation.py#L105). 
   - Look for the `generate` function within these files and apply any changes required for the new attack to work correctly.


## Adding a New Defense

1. **Prepare the Defense Class in `art` Format**
   - Ensure your defense class follows the `art` library structure and includes a `__call__` method.

2. Step 2+3 are the same as the new attack
3.  **Add the Defense to the `art-handler` Repository**
    - In the `attack_defense_map.json` file, add the new defense details as follows:
     ```json
     "14":{
             "name":"Counter_Samples",
             "class_name":"art_attacks_plugin.Counter_Samples.Counter_Samples",
             "type":"preprocessor",
             "description":"The implemetation of CounterSamples presented in the paper : https://arxiv.org/abs/2403.10562"
        }
       ```
   - Open the `setup.py` file in the `art-handler` repository, increase the version number, and rebuild the package:
     ```bash
     python setup.py bdist_wheel
     ```
   - Upload the package to Artifact Registry:
     ```bash
     twine upload --repository-url https://us-central1-python.pkg.dev/autodefenseml/art-handler/ dist/*
     ```
 4. **Import the defense Package in Code**
   - In the appropriate file (e.g., `attack_defense_validation.py`), import your defense package:
     ```python
     from <package_name> import *
     ```
5. **Make Additional Changes as Needed**
   - If your defense requires additional parameters or configurations, make the necessary modifications in `attack_defense_validation.py` (in the `pre_run` package) and `model_evaluation.py` (in the `eval` package). For example, add the hyperparameters and their ranges for optimizaiton [here](https://github.com/Offensive-AI-Lab/AutoDefenceML/blob/a8a3dfe07419b5853ba6b160054ba00c3dbfefeb/Backend/api/src/eval/evaluation/model_evaluation.py#L470).
   - Look for the `__call__` function within these files and apply any changes required for the new defense to work correctly.


   
     
