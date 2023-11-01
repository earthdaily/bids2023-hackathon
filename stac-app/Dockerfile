FROM mcr.microsoft.com/planetary-computer/python@sha256:e04d5237197a14f50450c111ef15c1de6b80115437424bcecf6959b85115d148 as common
USER root
RUN chown -R root /home/jovyan
SHELL ["conda", "run", "-n", "notebook", "/bin/bash", "-c"]
COPY requirements.txt .
RUN /srv/conda/envs/notebook/bin/python -m pip install -r requirements.txt
COPY . .

FROM common as dev
WORKDIR /code

FROM common as app
EXPOSE 8050
CMD ["conda", "run", "-n", "notebook", "--no-capture-output", "gunicorn", "--config", "gunicorn.conf.py", "app.app:server"]
