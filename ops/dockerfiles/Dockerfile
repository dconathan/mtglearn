FROM nvidia/cuda:11.3.1-runtime-centos8

RUN yum update -y && \ 
    yum install -y python39 gcc && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html && \
    python3 -m pip install transformers datasets ftfy
