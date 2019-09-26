FROM tensorflow/tensorflow:1.13.1-py3
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y xvfb
RUN apt-get install wget
RUN apt-get install unzip

EXPOSE 5005

RUN apt-get install -y git 

RUN pip install Pillow
RUN pip install Keras
RUN pip install moviepy
RUN pip install Pympler
RUN pip install mlagents_envs==0.9.2

RUN git clone https://github.com/koryakinp/mldriver-discrete-steering.git

WORKDIR /mldriver-discrete-steering

RUN wget https://github.com/koryakinp/MLDriver/releases/download/1.0/MLDriver_Linux_x86_64.zip
RUN mkdir environments
RUN unzip MLDriver_Linux_x86_64.zip -d environments/
RUN rm MLDriver_Linux_x86_64.zip
WORKDIR /mldriver-discrete-steering/environments
RUN rm -rf __MACOSX

WORKDIR /mldriver-discrete-steering

RUN chmod 755 runner.sh
ENTRYPOINT [ "./runner.sh" ]
CMD ["-c config1.py -e new"]