# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

EXPOSE 5002

WORKDIR /app
COPY . /app

# Install pip requirements
COPY req.txt .
RUN pip install -r req.txt


CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]

