import requests

# ✅ Chemin vers ton image locale (modifie si besoin)
image_path = "C:/Users/selfe/OneDrive/Bureau/Projet principal Stevia/backend_api/Data/STEVIAIA/Healthy/Stevia__90473.jpg"

# ✅ URL de l'API
url = "http://127.0.0.1:5000/predict"

# ✅ Données supplémentaires à envoyer
data = {
    "humidite": "45",    # % d'humidité ambiante
    "lumiere": "moyen"   # faible / moyen / forte
}

# ✅ Envoi de l’image + données
with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files, data=data)

# ✅ Résultat de l'IA
print("Statut de la réponse :", response.status_code)
print("Réponse de l'IA :")
print(response.json())
