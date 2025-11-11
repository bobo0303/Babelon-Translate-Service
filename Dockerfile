FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
# 設置工作目錄  
# WORKDIR /app  
# COPY . /app  

WORKDIR /mnt

# 更新系統並安裝必要的軟體包  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    tzdata libgl1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make unixodbc-dev curl python3-dev python3-distutils git wget libvulkan1 libfreeimage-dev gnupg2 \  
    && apt-get clean && rm -rf /var/lib/apt/lists/*  

# 安裝 Microsoft ODBC Driver 17 for SQL Server
RUN if ! [[ "8 9 10 11 12" == *"$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2 | cut -d '.' -f 1)"* ]]; then \
        echo "Debian $(grep VERSION_ID /etc/os-release | cut -d '"' -f 2 | cut -d '.' -f 1) is not currently supported."; \
        exit 1; \
    fi

RUN curl -sSL -O https://packages.microsoft.com/config/debian/$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2 | cut -d '.' -f 1)/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb

RUN apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && ACCEPT_EULA=Y apt-get install -y mssql-tools \
    && apt-get install -y libgssapi-krb5-2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 設置 SQL Server 工具路徑
ENV PATH="$PATH:/opt/mssql-tools/bin"  
    
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
# HUGGINGFACE_HUB_TOKEN
# hf auth login

# 該專案有寫 DB + Blob Storage 功能，請自行在 .env 裡面設定相關參數
# SAVE_AUDIO_TO_AZURE_BLOB
# AZURE_STORAGE_CONNECTION_STRING
# AZURE_STORAGE_CONTAINER_NAME

# DB -> alembic init -> alembic.ini 刪除 sqlalchemy.url 內容 -> alembic revision --autogenerate -m "init table" -> python3 init_job.py
 