from dataset import get_bts_data_list
data = get_bts_data_list("./brats_dataset")
print(f"Nombre de patients détectés : {len(data)}")
print(f"Exemple de chemin image : {data[0]['image']}")