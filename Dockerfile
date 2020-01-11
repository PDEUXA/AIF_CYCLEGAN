FROM tensorflow/tensorflow:latest-gpu-py3
#RUN apt-get update
#RUN apt-get install -y python3-pip
#RUN pip3 install --upgrade pip
COPY . /AIF_CycleGan
WORKDIR /AIF_CycleGan
#RUN pip install -r requirements.txt
RUN pip3 install tensorflow-datasets
#RUN pip3 install tensorflow-addons
RUN pip3 install matplotlib
RUN pip3 install argparse




