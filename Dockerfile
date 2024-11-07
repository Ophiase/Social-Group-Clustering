FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]
COPY . .
ENTRYPOINT ["conda", "run", "-n", "your_env_name", "python", "cli.py"]
