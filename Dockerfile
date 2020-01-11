FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
COPY . /AIF_CycleGan
WORKDIR /AIF_CycleGan
RUN pip install -r requirements.txt




