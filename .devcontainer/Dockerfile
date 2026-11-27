FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=ubuntu

RUN apt-get update \
    && apt-get install --yes --no-install-recommends sudo \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" \
        > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && rm -rf /var/lib/apt/lists/*


RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc"

USER root

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git and other ubuntu stuff
RUN sudo apt install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    git-lfs \
    wget \
    nano \
    eza \
    unzip \
    mesa-utils \
    libglfw3 \
    xterm \
    ripgrep \
    htop \
    bat \
    less \
    zsh

# python required dependencies
COPY requirements.txt /tmp/requirements.txt
COPY wheels /tmp/wheels

# install python3.13
# Ubuntu 26.04 ships no python3.13 package, so use a standalone build via uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13

# initialize virtual environment
ENV VIRTUAL_ENV=/python/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN uv venv --seed --python 3.13 "${VIRTUAL_ENV}"

# install python dependencies inside venv
# NOTE: dont use `python3 -m pip`!
#		`pip` resolves to `$VIRTUAL_ENV`/bin/pip,
# 		while the former does not (nessacerely)!
# requirements.txt references ./wheels relative to the working directory
WORKDIR /tmp
RUN pip install --upgrade pip setuptools wheel
RUN pip install --requirement /tmp/requirements.txt --find-links /tmp/wheels
RUN chown -R ${USERNAME} "${VIRTUAL_ENV}"


RUN export PATH="/home/vscode.local/bin:$PATH"
#RUN source ~/.bashrc   

# Switch from root to user
USER $USERNAME
WORKDIR /workspace

