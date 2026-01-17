from label_to_kaithi import label_to_kaithi
from mapping import map_kaithi_to_hindi

for k, v in label_to_kaithi.items():
    print(k, "→", v, "→", map_kaithi_to_hindi(v))
