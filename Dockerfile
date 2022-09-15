FROM nvcr.io/nvidia/pytorch:22.08-py3

COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT [ "/bin/bash" ]