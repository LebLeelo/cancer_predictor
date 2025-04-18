# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import traceback

app = Flask(__name__)

# Définir les noms des gènes (features)
gene_columns = [
    'gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_6', 'gene_7', 'gene_10', 
    'gene_11', 'gene_12', 'gene_13', 'gene_14', 'gene_17', 'gene_18', 'gene_19', 
    'gene_20', 'gene_21', 'gene_22', 'gene_23', 'gene_24', 'gene_25', 'gene_26', 
    'gene_27', 'gene_28', 'gene_29', 'gene_30', 'gene_31', 'gene_32', 'gene_33', 
    'gene_34', 'gene_35', 'gene_36', 'gene_37', 'gene_38', 'gene_39', 'gene_40', 
    'gene_41', 'gene_42', 'gene_43', 'gene_44', 'gene_45', 'gene_46', 'gene_47', 
    'gene_48', 'gene_49'
]

# Définir les noms des classes de cancer
cancer_types = {
    0: "  ",
    1: "Cancer du sein",
    2: "Cancer du rein",
    3: "Cancer du colon",
    4: "Cancer du poumon",
    5: "Cancer de la prostate"
}

# Variables globales pour le modèle et le scaler
model = None
scaler = None

def load_ml_components():
    """Charge le modèle et le scaler s'ils existent"""
    global model, scaler
    
    try:
        if os.path.exists('cancer_classifier_model.h5'):
            model = load_model('cancer_classifier_model.h5')
            print("Modèle chargé avec succès!")
        else:
            print("ATTENTION: Le fichier du modèle 'cancer_classifier_model.h5' n'existe pas!")
        
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            print("Scaler chargé avec succès!")
        else:
            print("ATTENTION: Le fichier du scaler 'scaler.pkl' n'existe pas!")
            
        return model is not None and scaler is not None
    
    except Exception as e:
        print(f"Erreur lors du chargement des composants ML: {e}")
        traceback.print_exc()
        return False

# Charger les composants avant de définir les routes
ml_ready = load_ml_components()

@app.route('/')
def home():
    return render_template('index.html', genes=gene_columns, ml_ready=ml_ready)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Le modèle ou le scaler n\'est pas chargé. Veuillez exécuter le script d\'entraînement d\'abord.'})
        
        # Récupérer les données du formulaire
        gene_values = []
        for gene in gene_columns:
            value = request.form.get(gene, '0')
            gene_values.append(float(value) if value else 0)
        
        # Préparer les données pour la prédiction
        input_data = np.array([gene_values])
        input_data_scaled = scaler.transform(input_data)
        
        # Faire la prédiction
        prediction = model.predict(input_data_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        probabilities = prediction[0] * 100  # Convertir en pourcentages
        
        # Préparer les résultats
        result = {
            'cancer_type': cancer_types[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'all_probabilities': {cancer_types[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
        
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/sample_data', methods=['GET'])
def sample_data():
    # Fournir des exemples de données pour chaque type de cancer
    samples = {
        "Cancer du sein": [2.325242483, 3.247092027, 8.174006857, 10.06505255, 7.487446152, 0, 0, 0, 2.759581973, 0, 0, 0, 0, 6.944553949, 2.951214764, 0, 7.163659738, 0, 0, 0, 0.566571641, 9.569891675, 0.566571641, 4.742653014, 6.921840937, 0.566571641, 7.769864285, 0, 12.05427192, 6.904568393, 0.566571641, 0, 9.73370853, 0, 10.08626894, 0, 0, 0.566571641, 2.759581973, 7.149066251, 9.382984925, 11.39379264, 11.64218259, 0],
        "Cancer du rein": [3.44619005, 3.620961697, 7.17191735, 9.796558821, 8.175245137, 1.972508859, 0, 1.972508859, 2.21896711, 0, 0, 0, 0, 8.617963278, 3.823779926, 0, 7.783574517, 0, 0, 0, 1.300006547, 9.492714449, 0.791689067, 4.459300459, 7.451277062, 0.791689067, 8.21273955, 5.230986874, 10.38215926, 4.094641454, 0, 0, 9.968537204, 0, 10.49974625, 0, 0, 0, 0, 7.292137545, 10.07309721, 8.763408255, 11.35489399, 0.791689067]
    }
    
    return jsonify(samples)

if __name__ == '__main__':
    # Exécution avec debug=False pour éviter le double chargement du modèle
    port = int(os.environ.get("PORT", 10000))  # Render utilise PORT ou 10000 par défaut
    app.run(host='0.0.0.0', port=port, debug=False)