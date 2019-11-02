FROM koryakinp/mlagents

RUN pipenv install Pillow
RUN pipenv install Keras
RUN pipenv install moviepy
RUN pipenv install Pympler
RUN pipenv install scikit-learn
RUN pipenv install tensorflow==1.13.1

RUN git clone https://github.com/koryakinp/mldriver-discrete-steering.git

WORKDIR /python-env/mldriver-discrete-steering

RUN wget https://github.com/koryakinp/MLDriver/releases/download/5.1/MLDriver_Linux_x86_64.zip
RUN mkdir environments
RUN unzip MLDriver_Linux_x86_64.zip -d environments/
RUN rm MLDriver_Linux_x86_64.zip
RUN rm -rf /python-env/mldriver-discrete-steering/environments/__MACOSX

RUN chmod 755 runner.sh

ENTRYPOINT [ "./runner.sh" ]
CMD ["-c config1.json -e new"]