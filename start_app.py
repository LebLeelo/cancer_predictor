# start_app.py
import os
import sys
import subprocess

def main():
    print("=== Démarrage de l'application de prédiction de cancer ===")
    
    # Vérifier si le modèle existe
    if not os.path.exists('cancer_classifier_model.h5') or not os.path.exists('scaler.pkl'):
        print("\nModèle ou scaler non trouvé. Voulez-vous entraîner le modèle maintenant? (o/n)")
        choice = input().strip().lower()
        
        if choice == 'o' or choice == 'oui':
            # Vérifier si le fichier de données existe
            if not os.path.exists('DNA_Dataset_Normalized.csv'):
                print("\nERREUR: Le fichier de données 'DNA_Dataset_Normalized.csv' n'existe pas.")
                print("Veuillez placer votre fichier de données dans le répertoire et réessayer.")
                sys.exit(1)
                
            print("\nEntraînement du modèle en cours...")
            try:
                # Exécuter le script d'entraînement
                subprocess.run([sys.executable, 'train_model.py'], check=True)
            except subprocess.CalledProcessError:
                print("\nERREUR: L'entraînement du modèle a échoué.")
                sys.exit(1)
        else:
            print("\nL'application sera lancée sans modèle disponible.")
    
    # Lancer l'application Flask
    print("\nDémarrage du serveur web...")
    subprocess.run([sys.executable, 'app.py'])

if __name__ == "__main__":
    main()