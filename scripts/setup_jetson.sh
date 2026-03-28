#!/bin/bash
# Jetson Orin Nano 4GB 초기 설정 스크립트
# JetPack 6.x 플래싱 후 실행

set -e

echo "=== Jetson Orin Nano 4GB Setup ==="

# 1. GUI 끄기 (500MB+ 메모리 절약)
echo "[1/5] Disabling GUI..."
sudo systemctl set-default multi-user.target

# 2. SWAP 확대 (8GB)
echo "[2/5] Configuring swap (8GB)..."
if [ ! -f /swapfile ] || [ "$(stat -c%s /swapfile 2>/dev/null)" -lt 8000000000 ]; then
    sudo swapoff /swapfile 2>/dev/null || true
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "  Swap configured"
else
    echo "  Swap already configured"
fi

# 3. 최대 성능 모드
echo "[3/5] Setting max performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# 4. Python 패키지 설치
echo "[4/5] Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    ultralytics \
    opencv-python \
    pyyaml \
    numpy

# 5. 네트워크 설정 (Zybo 직결용)
echo "[5/5] Configuring static IP for Zybo connection..."
CONN_NAME="zybo-direct"
if ! nmcli connection show "$CONN_NAME" &>/dev/null; then
    sudo nmcli connection add \
        type ethernet \
        con-name "$CONN_NAME" \
        ifname eth0 \
        ipv4.addresses 192.168.1.20/24 \
        ipv4.method manual
    echo "  Network connection '$CONN_NAME' created"
else
    echo "  Network connection '$CONN_NAME' already exists"
fi

echo ""
echo "=== Setup Complete ==="
echo "Reboot recommended: sudo reboot"
echo ""
echo "After reboot:"
echo "  1. Connect MIPI cameras"
echo "  2. Run: python3 scripts/test_cameras.py"
echo "  3. Run: python3 scripts/export_tensorrt.py"
echo "  4. Run: python3 fusion_server.py"
