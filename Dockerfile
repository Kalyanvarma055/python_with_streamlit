FROM python:3.12-slim
WORKDIR /docker_deploy
COPY app.py /docker_deploy/app.py
COPY requirements.txt /docker_deploy/requirements.txt
COPY accident.csv /docker_deploy/accident.csv
EXPOSE 8052
RUN pip install -r requirements.txt
CMD ["python","app.py"]