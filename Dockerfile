FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir torch torchvision gymnasium numpy
# Add any other libraries you used in inference.py

CMD ["python3", "inference.py"]
