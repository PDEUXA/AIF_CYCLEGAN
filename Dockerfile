FROM python:3.7-alpine
RUN mkdir /CycleGan
WORKDIR /CycleGan
COPY ./AIF_CycleGan /CycleGan/AIF_CycleGan
RUN pip install --upgrade pip
RUN pip install -r /CycleGan/requirements.txt

