#!/bin/bash
# ============================================
# Babelon 遷移匯入腳本
# 在【目標機器】上執行
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================"
echo " Babelon 遷移匯入工具"
echo "============================================"
echo ""

cd "$PROJECT_DIR"

# 檢查清單
echo -e "${BLUE}1. 檢查必要檔案...${NC}"

check_file() {
    if [ -f "$1" ] || [ -d "$1" ]; then
        echo -e "  ✅ $1"
        return 0
    else
        echo -e "  ❌ $1 ${RED}(缺少)${NC}"
        return 1
    fi
}

MISSING=0

check_file ".env" || MISSING=1
check_file ".env.notify" || MISSING=1
check_file "docker-compose.yml" || MISSING=1
check_file "Dockerfile" || MISSING=1
check_file "models/" || MISSING=1
check_file "vad/" || MISSING=1
check_file "scripts/start.sh" || MISSING=1
check_file "scripts/notify.sh" || MISSING=1

echo ""

if [ $MISSING -eq 1 ]; then
    echo -e "${RED}有檔案缺少，請先完成同步！${NC}"
    echo ""
    echo "缺少的檔案需要從來源機器同步："
    echo "  - .env, .env.notify: 使用 scp 複製"
    echo "  - models/, vad/: 使用 rsync 同步"
    echo ""
    exit 1
fi

echo -e "${GREEN}所有必要檔案已就緒！${NC}"
echo ""

# 檢查 Docker
echo -e "${BLUE}2. 檢查 Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "  ✅ Docker 已安裝: $(docker --version)"
else
    echo -e "  ❌ Docker 未安裝"
    echo ""
    echo "請先安裝 Docker："
    echo "  curl -fsSL https://get.docker.com | sh"
    echo "  sudo usermod -aG docker \$USER"
    exit 1
fi

if command -v docker compose &> /dev/null; then
    echo -e "  ✅ Docker Compose 已安裝"
else
    echo -e "  ❌ Docker Compose 未安裝"
    exit 1
fi

# 檢查 NVIDIA Docker
echo ""
echo -e "${BLUE}3. 檢查 NVIDIA Docker...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "  ✅ NVIDIA Docker 正常"
else
    echo -e "  ${YELLOW}⚠️  NVIDIA Docker 可能未設定${NC}"
    echo "  如果需要 GPU 支援，請安裝 nvidia-container-toolkit"
fi

# 建立必要目錄
echo ""
echo -e "${BLUE}4. 建立必要目錄...${NC}"
mkdir -p audio logs models
echo -e "  ✅ 目錄已建立"

# 設定腳本權限
echo ""
echo -e "${BLUE}5. 設定腳本權限...${NC}"
chmod +x scripts/*.sh
echo -e "  ✅ 權限已設定"

# 檢查 Discord Webhook（可選）
echo ""
echo -e "${BLUE}6. 檢查通知設定...${NC}"
if [ -f ".env.notify" ]; then
    source .env.notify
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        echo -e "  ✅ Discord Webhook 已設定"
    else
        echo -e "  ${YELLOW}⚠️  Discord Webhook 未設定（通知功能將停用）${NC}"
    fi
fi

# 更新 VM 名稱
echo ""
echo -e "${BLUE}7. 更新 VM 識別名稱...${NC}"
CURRENT_HOSTNAME=$(hostname)
read -p "請輸入此 VM 的識別名稱 [$CURRENT_HOSTNAME]: " VM_NAME
VM_NAME="${VM_NAME:-$CURRENT_HOSTNAME}"

if [ -f ".env.notify" ]; then
    sed -i "s/^VM_NAME=.*/VM_NAME=\"$VM_NAME\"/" .env.notify
    echo -e "  ✅ VM 名稱已更新為: $VM_NAME"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} 遷移準備完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "下一步："
echo "  1. 啟動服務: ./scripts/start.sh"
echo "  2. 安裝開關機通知（可選）: sudo ./scripts/install_monitor.sh"
echo ""
echo "首次啟動會："
echo "  - Build Docker Image"
echo "  - 執行完整測試"
echo "  - 發送 Discord 通知"
echo ""
read -p "是否現在啟動服務？[y/N] " START_NOW
if [[ "$START_NOW" =~ ^[Yy]$ ]]; then
    ./scripts/start.sh
fi
