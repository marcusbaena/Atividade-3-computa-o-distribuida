FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_MCA_orte_base_help_aggregate=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenmpi-dev \
        openmpi-bin \
        openssh-client \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && useradd --create-home --shell /bin/bash mpiuser \
    && mkdir -p /home/mpiuser/.ssh \
    && ssh-keygen -q -t ed25519 -N "" -f /home/mpiuser/.ssh/id_ed25519 \
    && cat /home/mpiuser/.ssh/id_ed25519.pub > /home/mpiuser/.ssh/authorized_keys \
    && printf "Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n" > /home/mpiuser/.ssh/config \
    && chmod 700 /home/mpiuser/.ssh \
    && chmod 600 /home/mpiuser/.ssh/authorized_keys /home/mpiuser/.ssh/config \
    && chown -R mpiuser:mpiuser /home/mpiuser/.ssh \
    && ssh-keygen -A

COPY requirements.txt /tmp/requirements.txt
RUN MPICC=mpicc pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/mpiuser/lab
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
