#!/bin/bash
# Deploy Deep Blue Web on Raspberry Pi
# Run as: sudo bash deploy.sh
set -e

echo "=== Deep Blue Web Deployment ==="

# 1. Install packages
echo "[1/8] Installing packages..."
apt-get update -qq
apt-get install -y -qq hostapd dnsmasq > /dev/null 2>&1

# Stop services during config
systemctl stop hostapd 2>/dev/null || true
systemctl stop dnsmasq 2>/dev/null || true

# 2. Install Flask in venv
echo "[2/8] Installing Flask..."
sudo -u enigma /home/enigma/yolo-env2/bin/pip install -q flask

# 3. Configure hostapd
echo "[3/8] Configuring Wi-Fi hotspot..."
cp /home/enigma/deep_blue_web/hostapd.conf /etc/hostapd/hostapd.conf

# Set DAEMON_CONF in hostapd defaults
if grep -q "^#DAEMON_CONF" /etc/default/hostapd 2>/dev/null; then
    sed -i 's|^#DAEMON_CONF.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd
elif ! grep -q "DAEMON_CONF" /etc/default/hostapd 2>/dev/null; then
    echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' >> /etc/default/hostapd
fi

# 4. Configure dnsmasq
echo "[4/8] Configuring DHCP..."
# Backup original
[ -f /etc/dnsmasq.conf.orig ] || cp /etc/dnsmasq.conf /etc/dnsmasq.conf.orig 2>/dev/null || true
cp /home/enigma/deep_blue_web/dnsmasq.conf /etc/dnsmasq.conf

# 5. Configure static IP on wlan0
echo "[5/8] Configuring static IP on wlan0..."
if ! grep -q "interface wlan0" /etc/dhcpcd.conf 2>/dev/null; then
    cat >> /etc/dhcpcd.conf << 'EOF'

# Deep Blue Web - Wi-Fi Hotspot
interface wlan0
    static ip_address=192.168.4.1/24
    nohook wpa_supplicant
EOF
fi

# 6. Configure mDNS (avahi)
echo "[6/8] Configuring mDNS (forklift.local)..."
if [ -f /etc/avahi/avahi-daemon.conf ]; then
    sed -i 's/^host-name=.*/host-name=forklift/' /etc/avahi/avahi-daemon.conf
    # If no host-name line exists, add it
    if ! grep -q "^host-name=" /etc/avahi/avahi-daemon.conf; then
        sed -i '/^\[server\]/a host-name=forklift' /etc/avahi/avahi-daemon.conf
    fi
fi
systemctl restart avahi-daemon 2>/dev/null || true

# 7. Install systemd service
echo "[7/8] Installing systemd service..."
cp /home/enigma/deep_blue_web/deep-blue-web.service /etc/systemd/system/
systemctl daemon-reload

# Disable old Deep Blue cron
echo "[8/8] Disabling old Deep Blue autostart..."
sudo -u enigma crontab -l 2>/dev/null | sed 's|^\(\* \* \* \* \* /home/enigma/deep_blue_start.sh\)|# \1|' | sudo -u enigma crontab -

# Kill old deep_blue.py if running
pkill -f "python3 /home/enigma/deep_blue.py" 2>/dev/null || true

# Enable and unmask services
systemctl unmask hostapd
systemctl enable hostapd
systemctl enable dnsmasq
systemctl enable deep-blue-web.service

# Set headless boot
systemctl set-default multi-user.target

# Start services
echo "Starting services..."
systemctl restart dhcpcd
sleep 2
systemctl start hostapd
systemctl start dnsmasq
systemctl start deep-blue-web.service

echo ""
echo "=== Deployment Complete ==="
echo "Wi-Fi SSID: deepBlue"
echo "Wi-Fi Pass: safety123"
echo "Dashboard:  http://forklift.local  (or http://192.168.4.1)"
echo "Login:      deepblue / matrix18"
echo ""
echo "Service status:"
systemctl is-active hostapd dnsmasq deep-blue-web.service
echo ""
echo "Reboot recommended: sudo reboot"
