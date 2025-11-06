FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
# 設置工作目錄  
# WORKDIR /app  
# COPY . /app  

WORKDIR /mnt

# 更新系統並安裝必要的軟體包  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    tzdata libgl1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make unixodbc-dev curl python3-dev python3-distutils git wget libvulkan1 libfreeimage-dev \  
    && apt-get clean && rm -rf /var/lib/apt/lists/*  

# 升級 pip  
RUN pip3 install --upgrade pip  
  
# 將 requirements.txt 複製到 Docker 映像中  
COPY requirements.txt /tmp/requirements.txt  
COPY whl/wjy3-1.8.2-py3-none-any.whl /tmp/wjy3-1.8.2-py3-none-any.whl  
RUN pip3 install -r /tmp/requirements.txt  
  

# 設置環境變量  
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV TZ=Asia/Taipei
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/llvm-10/lib:$LD_LIBRARY_PATH  
ENV PYTHONPATH=/mnt  

# 設置時區  
RUN ln -sf /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \  
    echo "Asia/Taipei" > /etc/timezone

# huggingface-cli login (要用 Gemma 要先登入 huggingface 要有 token)
# hf auth login

