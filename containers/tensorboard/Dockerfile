FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Install TensorBoard
RUN pip install --no-cache-dir tensorboard

# Expose the default TensorBoard port
EXPOSE 6006

# Command to run TensorBoard
CMD ["tensorboard", "--logdir", "/app/logs", "--host", "0.0.0.0"]