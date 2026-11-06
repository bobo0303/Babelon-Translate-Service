import requests

base_url = "http://localhost:80"
# 1) Call /get_prompt first
r = requests.get(f"{base_url}/get_prompt")
print("GET /get_prompt:", r.status_code, r.json())

import requests

url = "http://localhost:80/set_prompt"
data = {
    "prompts": "業說會, Micro LED, Monitor, QoQ, china car, FindARTs, DSBG, low temp, Tier, E&E, YoY, In-Cell Touch, Vertical, Firmware, AecoPost, DaaS, OLED, AmLED, Polarizer, Tartan Display, 達擎, ADP team, Legamaster, AVOCOR, RISEvision, JECTOR, SatisCtrl, 主管, Karl Storz, 光通訊, Schwarz, NATISIX, 友達, Pillar, 凌華, ComQi, AMSC, CPO, 流域, 共榮大會, TNFD, 小花蔓澤蘭, 邁向共生, CSR, 霄裡溪, AI 四支箭, paul, AUO, 彭双浪, 柯富仁."
}

resp = requests.post(url, data=data)
print(resp.status_code)
print(resp.json())