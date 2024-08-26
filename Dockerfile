# Dockerfile
FROM ermaker/keras

WORKDIR /app

COPY mnist_model.py /app/mnist_model.py

CMD ["python", "/app/mnist_model.py"]