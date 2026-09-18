FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# OpenMPI + Python + mpi4py (pacote do apt, compilado contra o mesmo OpenMPI) + SSH
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin libopenmpi-dev \
        python3 python3-pip python3-mpi4py \
        openssh-server openssh-client \
        iputils-ping nano \
    && rm -rf /var/lib/apt/lists/*

# Usuario que executa o MPI
RUN useradd -m -s /bin/bash mpiuser && mkdir -p /var/run/sshd

# Chave SSH sem senha compartilhada por todos os nos (mesma imagem => mesma chave)
USER mpiuser
RUN mkdir -p /home/mpiuser/.ssh \
    && ssh-keygen -t ed25519 -N "" -f /home/mpiuser/.ssh/id_ed25519 \
    && cp /home/mpiuser/.ssh/id_ed25519.pub /home/mpiuser/.ssh/authorized_keys \
    && printf "Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n" > /home/mpiuser/.ssh/config \
    && chmod 700 /home/mpiuser/.ssh \
    && chmod 600 /home/mpiuser/.ssh/*

COPY --chown=mpiuser:mpiuser hosts /home/mpiuser/hosts

USER root
# Evita erro de memoria compartilhada (CMA) do OpenMPI dentro de containers
RUN echo "btl_vader_single_copy_mechanism = none" >> /etc/openmpi/openmpi-mca-params.conf

EXPOSE 22
CMD ["bash", "-c", "service ssh start && sleep infinity"]
