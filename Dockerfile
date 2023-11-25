FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f env.yml

RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

SHELL ["/bin/bash", "--login", "-c"]

CMD ["python", "test.py"]

