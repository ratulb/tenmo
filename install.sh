#!/usr/bin/bash
clear
curl -fsSL https://pixi.sh/install.sh | sh
. source /root/.bashrc
git clone https://github.com/ratulb/tenmo
cd tenmo
pixi shell
clear
