#!/bin/bash
# ============================================
# Install Babelon Monitor systemd service
# 安裝開機/關機通知服務
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$SCRIPT_DIR/babelon-monitor.service"
SYSTEMD_DIR="/etc/systemd/system"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================"
echo " Babelon Monitor 安裝程式"
echo "============================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}需要 sudo 權限來安裝 systemd 服務${NC}"
    echo "請執行: sudo $0"
    exit 1
fi

# Make scripts executable
echo "設置腳本執行權限..."
chmod +x "$SCRIPT_DIR/notify.sh"
chmod +x "$SCRIPT_DIR/health_monitor.sh"
chmod +x "$SCRIPT_DIR/post_build_test.sh"
chmod +x "$SCRIPT_DIR/start.sh"
chmod +x "$SCRIPT_DIR/stop.sh"

# Update service file with correct user
CURRENT_USER=$(logname 2>/dev/null || echo "aioproomadmin")
echo "檢測到使用者: $CURRENT_USER"

# Create a customized service file
cat > "$SYSTEMD_DIR/babelon-monitor.service" << EOF
[Unit]
Description=Babelon Translation Service Monitor
After=network.target docker.service
Wants=docker.service
Before=shutdown.target reboot.target halt.target
DefaultDependencies=no

[Service]
Type=oneshot
RemainAfterExit=yes
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$SCRIPT_DIR/health_monitor.sh startup
ExecStop=/bin/bash -c '$SCRIPT_DIR/health_monitor.sh shutdown; sleep 2'
TimeoutStartSec=300
TimeoutStopSec=90
KillMode=none

[Install]
WantedBy=multi-user.target
EOF

echo "已建立 systemd 服務檔案"

# Reload systemd
echo "重新載入 systemd..."
systemctl daemon-reload

# Enable service
echo "啟用服務..."
systemctl enable babelon-monitor.service

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} 安裝完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "服務已安裝並啟用，系統開機/關機時會自動發送通知"
echo ""
echo "管理命令:"
echo "  sudo systemctl status babelon-monitor   # 查看狀態"
echo "  sudo systemctl start babelon-monitor    # 手動啟動（發送健康通知）"
echo "  sudo systemctl stop babelon-monitor     # 手動停止（發送關機通知）"
echo "  sudo systemctl disable babelon-monitor  # 停用開機通知"
echo ""
echo "測試通知:"
echo "  ./scripts/notify.sh test"
echo ""
