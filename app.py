from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/predict": {"origins": "*"}})

model = tf.keras.models.load_model("model_stevia.h5")
class_names = ['Diseased', 'Healthy']

def estimer_age(image_pil):
    gray_image = image_pil.convert('L')
    sharpness = gray_image.filter(ImageFilter.FIND_EDGES)
    edges = np.array(sharpness).mean()
    image_np = np.array(image_pil)
    moyenne_verte = np.mean(image_np[:, :, 1])

    if moyenne_verte > 150 and edges > 20:
        return "1 mois"
    elif 100 < moyenne_verte <= 150:
        return "3 mois"
    else:
        return "5 mois"

def calculer_maturite(age):
    return "Oui, la plante est mature." if age != "1 mois" else "Non, elle est encore jeune."

def calculer_chance_survie(maladie):
    return "Élevée (90%)" if maladie == "Healthy" else "Faible (30%)"

def ajustement_nutritif(maladie):
    if maladie == "Diseased":
        return "Oui. Apport recommandé : fertilisant organique riche en azote + vérifier le pH du sol (entre 6 et 7)."
    return "Pas nécessaire pour le moment."

def risque_pour_cultures(maladie):
    return "Oui. Il est recommandé d'isoler cette plante pour éviter la propagation." if maladie == "Diseased" else "Aucun risque détecté."

def generer_conseils(etat, age, humidite, lumiere):
    conseils = []

    if etat == "Diseased":
        if age == "1 mois":
            conseils += [
                "Isoler la plante",
                "Éviter les traitements chimiques",
                "Utiliser un antifongique doux"
            ]
        elif age == "3 mois":
            conseils += [
                "Appliquer un fongicide naturel",
                "Couper les feuilles abîmées",
                "Contrôler l’humidité ambiante"
            ]
        else:
            conseils += [
                "Réduire l’arrosage",
                "Repenser la fertilisation",
                "Retirer les feuilles anciennes"
            ]
    else:
        if age == "1 mois":
            conseils += [
                "Arroser légèrement 2 fois/semaine",
                "Éviter le soleil direct",
                "Contrôler l’apparition de taches"
            ]
        elif age == "3 mois":
            conseils += [
                "Maintenir une humidité modérée",
                "Éclairage doux recommandé",
                "Nettoyer les feuilles avec un chiffon humide"
            ]
        else:
            conseils += [
                "Surveiller le jaunissement naturel",
                "Éviter l’eau stagnante au pied",
                "Ajouter un peu d’engrais organique"
            ]

    if humidite:
        try:
            h = float(humidite)
            if h < 30:
                conseils.append("Humidité trop basse : brumisez légèrement")
            elif h > 80:
                conseils.append("Humidité trop élevée : pensez à aérer davantage")
        except:
            conseils.append("Valeur d'humidité incorrecte")

    if lumiere:
        l = lumiere.lower()
        if l == "faible":
            conseils.append("Lumière insuffisante : rapprocher d’une fenêtre")
        elif l == "forte":
            conseils.append("Lumière trop forte : filtrer avec un rideau léger")

    return conseils

@app.route('/')
def home():
    return "SteviaAI API is running ✅"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")
    age_estime = estimer_age(image)

    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = predictions.argmax()
    disease_label = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    humidite = request.form.get("humidite")
    lumiere = request.form.get("lumiere")

    response = {
        "maladie": disease_label,
        "confiance": round(confidence * 100, 2),
        "age_estime": age_estime,
        "maturite": calculer_maturite(age_estime),
        "chance_survie": calculer_chance_survie(disease_label),
        "ajustement_nutritif": ajustement_nutritif(disease_label),
        "risque_cultures": risque_pour_cultures(disease_label),
        "humidite": "60–70%",
        "lumiere": "Lumière indirecte forte",
        "conseils": generer_conseils(disease_label, age_estime, humidite, lumiere)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
