FROM python:3.13-bullseye
# FROM gcr.io/distroless/python3
# Author: Gabin Vrillault
# mail: gabin[dot]vrillault[at]ecole[dot]ensicaen[dot]fr
# Date: 2025-06-21

# Description: Dockerfile for the counterfactuals module.

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN chmod +x run_app.sh

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port that Streamlit uses
EXPOSE 8501

CMD [ "./run_app.sh" ]


