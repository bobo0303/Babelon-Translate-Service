#!/bin/bash
# ============================================
# Babelon 遷移匯出腳本
# 在【來源機器】上執行
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXPORT_DIR="${1:-/tmp/babelon_export}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================"
echo " Babelon 遷移匯出工具"
echo "============================================"
echo ""

cd "$PROJECT_DIR"

# 建立匯出目錄
mkdir -p "$EXPORT_DIR"

echo -e "${BLUE}1. 匯出敏感設定檔...${NC}"
cp -v .env "$EXPORT_DIR/.env" 2>/dev/null || echo "  .env 不存在"
cp -v .env.notify "$EXPORT_DIR/.env.notify" 2>/dev/null || echo "  .env.notify 不存在"

echo ""
echo -e "${BLUE}2. 產生 rsync 命令...${NC}"

# 取得目標機器資訊
read -p "請輸入目標機器 IP 或主機名稱: " TARGET_HOST
read -p "請輸入目標機器使用者名稱 [$(whoami)]: " TARGET_USER
TARGET_USER="${TARGET_USER:-$(whoami)}"
read -p "請輸入目標機器專案路徑 [$PROJECT_DIR]: " TARGET_PATH
TARGET_PATH="${TARGET_PATH:-$PROJECT_DIR}"

# 產生 rsync 命令檔
cat > "$EXPORT_DIR/sync_commands.sh" << EOF
#!/bin/bash
# ============================================
# Babelon 同步命令
# 在【來源機器】上執行這些命令
# ============================================

TARGET="${TARGET_USER}@${TARGET_HOST}:${TARGET_PATH}"

echo "=== 步驟 1: 同步程式碼（透過 Git）==="
echo "在目標機器上執行："
echo "  git clone <your-repo-url> ${TARGET_PATH}"
echo "  或 git pull（如果已存在）"
echo ""

echo "=== 步驟 2: 同步模型檔案 (約 15GB) ==="
echo "rsync -avz --progress ${PROJECT_DIR}/models/ \${TARGET}/models/"
echo ""

echo "=== 步驟 3: 同步 VAD 模型 ==="
echo "rsync -avz --progress ${PROJECT_DIR}/vad/ \${TARGET}/vad/"
echo ""

echo "=== 步驟 4: 同步音檔（可選，約 39GB）==="
echo "rsync -avz --progress ${PROJECT_DIR}/audio/ \${TARGET}/audio/"
echo ""

echo "=== 步驟 5: 複製設定檔 ==="
echo "scp ${EXPORT_DIR}/.env \${TARGET}/.env"
echo "scp ${EXPORT_DIR}/.env.notify \${TARGET}/.env.notify"
echo ""

# 實際執行的一鍵命令
echo "============================================"
echo " 一鍵同步（不含音檔）"
echo "============================================"
cat << 'SYNC_CMD'
# 複製以下命令執行：

rsync -avz --progress ${PROJECT_DIR}/models/ ${TARGET}/models/ && \\
rsync -avz --progress ${PROJECT_DIR}/vad/ ${TARGET}/vad/ && \\
scp ${EXPORT_DIR}/.env ${TARGET}/.env && \\
scp ${EXPORT_DIR}/.env.notify ${TARGET}/.env.notify && \\
echo "同步完成！"
SYNC_CMD

EOF

# 替換變數
sed -i "s|\${TARGET}|${TARGET_USER}@${TARGET_HOST}:${TARGET_PATH}|g" "$EXPORT_DIR/sync_commands.sh"
sed -i "s|\${PROJECT_DIR}|${PROJECT_DIR}|g" "$EXPORT_DIR/sync_commands.sh"
sed -i "s|\${EXPORT_DIR}|${EXPORT_DIR}|g" "$EXPORT_DIR/sync_commands.sh"

chmod +x "$EXPORT_DIR/sync_commands.sh"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} 匯出完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "匯出目錄: $EXPORT_DIR"
echo ""
echo "檔案清單:"
ls -la "$EXPORT_DIR/"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo "1. 在目標機器上 git clone 專案"
echo "2. 執行同步命令: bash $EXPORT_DIR/sync_commands.sh"
echo "3. 在目標機器上執行: ./scripts/migrate_import.sh"
echo ""
