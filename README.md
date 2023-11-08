# MlOps

This repo is a project based off New York taxi data to practice Mlops. The project is based off MlOps Zoomcamp. Each directory focuses on an aspect of MLOps

## Experiment Tracking

Experiment tracking  is a best practice in machine learning workflows, promoting transparency, reproducibility, and efficiency in the development and deployment of models. For experiment tracking we used MLflow for this project. MLflow allowed the tracking of different models experiments and automating the tuning of the model. With a tool like mlflow it also allows others on the team to have insight into the model experimints.

The mlflow tracking server was setup and ran on an aws ec2 for this project. The backend store was a postgres database on rds and the artifact store used was s3.

## Orchestration

For orchestration Prefect was used to create an MLOps workflow.Using Prefect a MLOps pipeline was created to automate training and gave observability into the flow of the pipeline. With this pipeline the model training was automated.

## Model Deployment

For model deployment online and offline techniques were used. For streaming deployment AWS Lambda and Kinesis were used. The model was also deployed as web service using Flask that could be used in a Kubernetes cluster for example. Also the model was deployed using the batching technique. 

## Monitoring

Lastly monitoring was implemented to give obeservability into our model and to detect anomalys in the model and data. The monitoring web services used were Prometheus, Grafana, and Evidently.

