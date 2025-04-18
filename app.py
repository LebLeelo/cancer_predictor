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
        "Cancer du rein": [2.98719351, 2.67730527, 7.134118239, 9.689576978, 6.186092444, 0, 1.05088474, 0, 2.402503901, 0, 0.619084288, 0, 0, 9.157862272, 1.652417657, 4.883928676, 7.743427449, 0, 0, 0.619084288, 6.698968529, 8.962479616, 5.490066254, 4.792032803, 0, 3.676933079, 6.310634592, 4.740037066, 14.02623813, 8.788065751, 0.619084288, 0, 9.059858169, 0 ,9.648744047, 0, 0, 0, 0.619084288, 8.450695772, 9.968767577, 10.95525758, 11.26114292, 0.619084288],
        "Cancer du colon": [1.373787394, 1.395995039, 6.359876038, 9.346245558, 5.463046938, 0, 0.852638117, 0, 3.468909092, 0, 0, 0, 8.933433956, 9.130902563, 1.933685934, 3.843602022, 9.147398289, 0, 0, 0.488412176, 0.488412176, 9.070561067, 5.716203506, 4.717330957, 8.022728442, 0, 9.493597302, 0, 15.73178306, 10.62751565, 2.330271553, 0, 9.493016802, 5.514816607, 8.452508953, 0, 0, 0.488412176, 0, 5.071891353, 8.828663313, 9.729115967, 11.42331029, 1.143197476],
        "Cancer du poumon": [3.590338533, 3.033986754, 6.29573037, 10.12518097, 8.186317317, 0, 0.578117532, 1.309525466, 1.309525466, 0, 0, 0, 0, 8.73583692, 4.503704912, 0, 9.298935484, 0, 0, 0, 5.94535019, 10.19295852, 5.18925132, 6.722370388, 0.578117532, 0, 6.767257526, 2.789562336, 15.29650043, 8.732496468, 2.789562336, 0, 9.729124466, 0, 10.05873589, 0, 0, 0.989647677, 2.442465835, 6.933596242, 9.535765265, 9.581029181, 11.0250011, 0.578117532],
        "Cancer de la prostate": [3.849068511, 3.178077197, 7.0284918, 10.65522659, 7.624583467, 0, 0, 0.659924558, 2.495797478, 0, 0, 0, 0.659924558, 11.68011774, 2.568129417, 3.044499039, 8.510522875, 0, 0, 0, 0.367371066, 8.058863031, 5.627428825, 5.629289584, 0, 0.659924558, 4.44552146, 2.339222918, 12.66018536, 7.294914771, 0.659924558, 0, 9.856945241, 1.731270148, 11.76106915, 0, 0, 0, 0, 3.782513408, 11.63043551, 10.29053729, 11.62382206, 0.903115418]
    }
    
    return jsonify(samples)

if __name__ == '__main__':
    # Exécution avec debug=False pour éviter le double chargement du modèle
    port = int(os.environ.get("PORT", 10000))  # Render utilise PORT ou 10000 par défaut
    app.run(host='0.0.0.0', port=port, debug=False)




    