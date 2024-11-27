FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "social-group-clustering", "/bin/bash", "-c"]
COPY . .
ENTRYPOINT ["conda", "run", "-n", "social-group-clustering", "python", "cli.py"]
