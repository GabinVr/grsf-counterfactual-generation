FROM python:3.14.0b2-bookworm
# Author: Gabin Vrillault
# mail: gabin[dot]vrillault[at]ecole[dot]ensicaen[dot]fr
# Date: 2025-06-21

# Description: Dockerfile for the counterfactuals module.

# This module was written with the help of Claude 3.5 Sonnet 

WORKDIR /app

COPY * /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x run_app.sh
# Expose the port that Streamlit uses
EXPOSE 8501

CMD [ "./run_app.sh" ]


