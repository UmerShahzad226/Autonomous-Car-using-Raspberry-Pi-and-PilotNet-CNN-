#!/bin/bash
# Wait for system to fully boot
sleep 15

nmcli device disconnect wlan0
sleep 2

sudo pigpiod
sleep 2

nmcli connection up "Hotspot"
sleep 5

# Start RC car server
cd /home/rccar/RC_Car
sudo python app.py
