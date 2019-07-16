FROM tensorflow/tensorflow:1.13.1-gpu-py3
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y xvfb

EXPOSE 5005

RUN apt-get install -y git 

RUN pip install Pillow
RUN pip install Keras
RUN pip install moviepy
RUN pip install Pympler

RUN git clone https://github.com/koryakinp/mldriver-discrete-steering.git
WORKDIR /mldriver-discrete-steering
COPY environments environments
RUN pip install -e .
RUN mkdir summaries
RUN chmod 755 runner.sh
ENTRYPOINT ["./runner.sh"]
CMD ["-e new"]