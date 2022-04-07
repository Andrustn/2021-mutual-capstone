FROM public.ecr.aws/lambda/python@sha256:8dc4d4145c1a05ed727b2e57e611c302c7f7188b406ba0934a612166458e524e as build
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

#OVERWRITE THE FIRST VALUE WITH YOUR PRIMARYFILE.py. KEEP THE "./"
COPY test.py ./

#OVERWRITE THE FOLLOWING VALUE WITH YOUR PRIMARYFILE.MAINMETHOD
CMD [ "test.handler" ]
