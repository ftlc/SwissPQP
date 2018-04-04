##Switzerland PQP 


Setting up Python Virtual Environment (to avoid issues with different environments across different systems)

  Assuming you have python and pip installed.

1.  Install virtualenv and initialize the environment
  ```
  pip install virtualenv
  virtualenv venv 
  ```
  The name is arbitraty. venv is just convenient and is already in the `.gitignore`

2. Activate the virtual environment
  ```
  source venv/bin/activate
  ```

  This gives us a localized environment with only the local packages installed. By default python packages installed globally are not available in the venv

3. Install the dependencies (locally)
  ```
  pip install -r requirements.txt
  ```

  Any time we need a new dependency simply add it to the requirements file. 
  It's best practice to use `pip freeze` as that will also specify versions
