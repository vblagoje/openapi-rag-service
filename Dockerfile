FROM deepset/haystack:base-v2.7.0

# Copy files to the root directory (Don't change)
# Enables smooth GitHub runner integration
# GitHub runner that spins up a docker container sets the working dir to /github/workspace and mounts
# the project directory to /github/workspace overriding main.py and other files.
# If you know a better way to do this, please let me know
COPY main.py /
COPY requirements.txt /

# Install required packages
RUN pip install --no-cache-dir -r /requirements.txt

# Set the entrypoint
ENTRYPOINT ["python", "/main.py"]
