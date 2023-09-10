FROM python:3.9
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY src src
COPY model model
COPY main.py main.py

EXPOSE 8888

ENTRYPOINT [ "python" ]

CMD ["main.py"]
