FROM tiangolo/uwsgi-nginx-flask:python3.6
LABEL maintainer="Simranjeet Singh <simranjeetsingh1497@gmail.com>"

COPY ./app_dash/requirements.txt /tmp/

RUN pip install -U pip && pip install -r /tmp/requirements.txt

ENV NGINX_WORKER_PROCESSES auto

HEALTHCHECK CMD curl --fail http://localhost:80/ || exit 1