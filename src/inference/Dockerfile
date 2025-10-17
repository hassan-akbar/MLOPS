# Base image for AWS Lambda (Python 3.11)
FROM public.ecr.aws/lambda/python:3.11

# Set working directory inside the Lambda container
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL inference code (from the context root)
# We'll copy everything from ../inference relative to Docker build context
COPY . .

# Set the Lambda entry point (your handler)
CMD ["lambda_function.lambda_handler"]
