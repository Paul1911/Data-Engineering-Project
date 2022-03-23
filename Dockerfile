FROM tiangolo/uwsgi-nginx-flask:python3.6
LABEL maintainer="Simranjeet Singh <simranjeetsingh1497@gmail.com>"

COPY ./app_dash/requirements.txt /tmp/
#COPY ./app_dash /app
#COPY ./dags/helpfiles/ml_pipeline_config.py /app/ml_pipeline_config.py

RUN pip install -U pip && pip install -r /tmp/requirements.txt
#RUN 

ENV NGINX_WORKER_PROCESSES auto