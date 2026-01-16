from src.mapping import map_kaithi_to_hindi

kaithi_text = "𑂎𑂯𑂧"
hindi_text = map_kaithi_to_hindi(kaithi_text)

print("Kaithi Input :", kaithi_text)
print("Hindi Output :", hindi_text)
