# Web_Application

This folder houses the web interface for the model.

## Instructions to develop "Web_Application"

- Create a feature branch for the feature you are developing (based on the issue on GitLab)


- Develop, use PyCharm preferably. Activate a virtual environment with the requirements.txt file in the project.


- Install pre-commit hooks to lint code when committing code. In terminal with the main project directory as working
  directory,
  type `pre-commit install`.


- Before pushing, make sure the Docker container runs. With location of Web_Application in terminal:
    - `docker build -t web_application:latest .`
    - `docker run -p 8080:8080 web_application:latest`
    - Make sure the functionality implemented is observable and all previous functionality is retained.


- Push to the feature branch created


- Raise merge request for peer review. Assign a reviewer for your code.


- After the reviewer has approved the merge request, rebase and pull into master.


