FROM tensorflow/tensorflow:1.13.1-py3

RUN apt-get install -y xvfb

EXPOSE 5005

RUN apt-get install -y git

RUN pip install Pillow

RUN git clone https://github.com/koryakinp/mldriver-discrete-steering.git
WORKDIR /mldriver-discrete-steering
COPY environments environments
RUN pip install -e .