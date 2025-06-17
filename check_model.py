import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import time
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === WRAPPER SAVEDMODEL (copié de votre code) ===
class SavedModelWrapper:
    """Wrapper pour utiliser un SavedModel comme un modèle Keras"""
    
    def __init__(self, savedmodel_path):
        self.model = tf.saved_model.load(savedmodel_path)
        self.serving_fn = self.model.signatures['serving_default']
        
        # Analyser la signature pour comprendre les entrées/sorties
        self.input_key = list(self.serving_fn.structured_input_signature[1].keys())[0]
        self.output_key = list(self.serving_fn.structured_outputs.keys())[0]
    
    def predict(self, x, verbose=0):
        """Méthode predict compatible avec Keras"""
        # Convertir l'entrée au bon format
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Faire la prédiction avec la signature
        input_dict = {self.input_key: x}
        output = self.serving_fn(**input_dict)
        
        # Extraire la sortie
        result = output[self.output_key].numpy()
        return result

# === FONCTIONS DE CHARGEMENT POUR CHAQUE TYPE ===
def load_keras_model():
    """Charge le modèle .keras"""
    try:
        model = tf.keras.models.load_model("skin_cancer_model.keras", compile=False)
        return model, "Keras (.keras)"
    except Exception as e:
        return None, f"Erreur .keras: {str(e)[:100]}"

def load_h5_model():
    """Charge le modèle .h5"""
    try:
        model = tf.keras.models.load_model("skin_cancer_model.h5", compile=False)
        return model, "H5 (.h5)"
    except Exception as e:
        return None, f"Erreur .h5: {str(e)[:100]}"

def load_savedmodel_keras():
    """Charge le SavedModel en tant que modèle Keras"""
    try:
        model = tf.keras.models.load_model("skin_cancer_model_savedmodel")
        return model, "SavedModel Keras"
    except Exception as e:
        return None, f"Erreur SavedModel Keras: {str(e)[:100]}"

def load_savedmodel_wrapper():
    """Charge le SavedModel avec wrapper"""
    try:
        model = SavedModelWrapper("skin_cancer_model_savedmodel")
        return model, "SavedModel Wrapper"
    except Exception as e:
        return None, f"Erreur SavedModel Wrapper: {str(e)[:100]}"

# === GÉNÉRATION DE DONNÉES DE TEST ===
def generate_test_data(num_samples=100):
    """Génère des données de test synthétiques"""
    np.random.seed(42)  # Pour la reproductibilité
    
    # Générer des images de test (64x64x3)
    test_images = np.random.rand(num_samples, 64, 64, 3).astype(np.float32)
    
    # Générer des labels de test (0-6 pour 7 classes)
    test_labels = np.random.randint(0, 7, num_samples)
    
    return test_images, test_labels

def load_real_test_data():
    """Essaie de charger de vraies données de test si disponibles"""
    # Vous pouvez modifier cette fonction pour charger vos vraies données
    st.info("📁 Recherche de données de test réelles...")
    
    # Vérifier si vous avez un dossier de test
    if os.path.exists("test_data") or os.path.exists("validation_data"):
        st.info("✅ Dossier de données trouvé, mais chargement non implémenté")
        st.info("💡 Utilisation de données synthétiques pour la démo")
    
    return generate_test_data()

# === ÉVALUATION DES MODÈLES ===
def evaluate_model(model, model_name, test_images, test_labels):
    """Évalue un modèle sur les données de test"""
    try:
        st.info(f"🔄 Évaluation de {model_name}...")
        
        # Mesurer le temps de prédiction
        start_time = time.time()
        predictions = model.predict(test_images, verbose=0)
        inference_time = time.time() - start_time
        
        # Obtenir les classes prédites
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculer les métriques
        accuracy = accuracy_score(test_labels, predicted_classes)
        
        # Temps moyen par prédiction
        avg_time_per_prediction = inference_time / len(test_images) * 1000  # en ms
        
        # Confiance moyenne
        confidence_scores = np.max(predictions, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        # Nombre de paramètres (si disponible)
        try:
            if hasattr(model, 'count_params'):
                num_params = model.count_params()
            elif hasattr(model, 'layers'):
                num_params = sum([layer.count_params() for layer in model.layers if hasattr(layer, 'count_params')])
            else:
                num_params = "N/A"
        except:
            num_params = "N/A"
        
        return {
            'Modèle': model_name,
            'Accuracy': f"{accuracy:.4f}",
            'Accuracy (%)': f"{accuracy*100:.2f}%",
            'Temps total (s)': f"{inference_time:.3f}",
            'Temps/pred (ms)': f"{avg_time_per_prediction:.2f}",
            'Confiance moy.': f"{avg_confidence:.4f}",
            'Nb paramètres': num_params,
            'Status': '✅ OK'
        }
        
    except Exception as e:
        return {
            'Modèle': model_name,
            'Accuracy': 'N/A',
            'Accuracy (%)': 'N/A',
            'Temps total (s)': 'N/A',
            'Temps/pred (ms)': 'N/A',
            'Confiance moy.': 'N/A',
            'Nb paramètres': 'N/A',
            'Status': f'❌ {str(e)[:50]}'
        }

# === INTERFACE STREAMLIT ===
def main():
    st.title("📊 Comparaison des modèles de détection de cancer de la peau")
    st.markdown("Cette application compare les performances de tous vos modèles disponibles.")
    
    # === Diagnostic des fichiers ===
    st.subheader("📁 Fichiers disponibles")
    files_to_check = [
        "skin_cancer_model.h5",
        "skin_cancer_model.keras", 
        "skin_cancer_model_savedmodel"
    ]
    
    available_models = []
    for file in files_to_check:
        if os.path.exists(file):
            if os.path.isdir(file):
                size_info = "dossier"
            else:
                size_mb = os.path.getsize(file) / (1024 * 1024)
                size_info = f"{size_mb:.1f} MB"
            st.write(f"✅ {file} ({size_info})")
            available_models.append(file)
        else:
            st.write(f"❌ {file} non trouvé")
    
    if not available_models:
        st.error("❌ Aucun modèle trouvé!")
        return
    
    # === Configuration des tests ===
    st.subheader("⚙️ Configuration des tests")
    
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.slider("Nombre d'échantillons de test", 10, 500, 100)
    with col2:
        use_real_data = st.checkbox("Utiliser de vraies données (si disponibles)", False)
    
    # === Bouton de lancement ===
    if st.button("🚀 Lancer la comparaison"):
        
        # Chargement des données de test
        if use_real_data:
            test_images, test_labels = load_real_test_data()
        else:
            test_images, test_labels = generate_test_data(num_samples)
        
        st.success(f"📊 Données de test générées: {len(test_images)} échantillons")
        
        # === Évaluation de tous les modèles ===
        results = []
        
        # 1. Modèle .keras
        if "skin_cancer_model.keras" in available_models:
            model, status = load_keras_model()
            if model is not None:
                result = evaluate_model(model, "Keras (.keras)", test_images, test_labels)
                results.append(result)
                del model  # Libérer la mémoire
            else:
                results.append({
                    'Modèle': 'Keras (.keras)',
                    'Status': f'❌ {status}',
                    'Accuracy': 'N/A', 'Accuracy (%)': 'N/A',
                    'Temps total (s)': 'N/A', 'Temps/pred (ms)': 'N/A',
                    'Confiance moy.': 'N/A', 'Nb paramètres': 'N/A'
                })
        
        # 2. Modèle .h5
        if "skin_cancer_model.h5" in available_models:
            model, status = load_h5_model()
            if model is not None:
                result = evaluate_model(model, "H5 (.h5)", test_images, test_labels)
                results.append(result)
                del model
            else:
                results.append({
                    'Modèle': 'H5 (.h5)',
                    'Status': f'❌ {status}',
                    'Accuracy': 'N/A', 'Accuracy (%)': 'N/A',
                    'Temps total (s)': 'N/A', 'Temps/pred (ms)': 'N/A',
                    'Confiance moy.': 'N/A', 'Nb paramètres': 'N/A'
                })
        
        # 3. SavedModel Keras
        if "skin_cancer_model_savedmodel" in available_models:
            model, status = load_savedmodel_keras()
            if model is not None:
                result = evaluate_model(model, "SavedModel Keras", test_images, test_labels)
                results.append(result)
                del model
            else:
                results.append({
                    'Modèle': 'SavedModel Keras',
                    'Status': f'❌ {status}',
                    'Accuracy': 'N/A', 'Accuracy (%)': 'N/A',
                    'Temps total (s)': 'N/A', 'Temps/pred (ms)': 'N/A',
                    'Confiance moy.': 'N/A', 'Nb paramètres': 'N/A'
                })
            
            # 4. SavedModel Wrapper
            model, status = load_savedmodel_wrapper()
            if model is not None:
                result = evaluate_model(model, "SavedModel Wrapper", test_images, test_labels)
                results.append(result)
                del model
            else:
                results.append({
                    'Modèle': 'SavedModel Wrapper',
                    'Status': f'❌ {status}',
                    'Accuracy': 'N/A', 'Accuracy (%)': 'N/A',
                    'Temps total (s)': 'N/A', 'Temps/pred (ms)': 'N/A',
                    'Confiance moy.': 'N/A', 'Nb paramètres': 'N/A'
                })
        
        # === Affichage des résultats ===
        st.subheader("📊 Résultats de la comparaison")
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # === Analyse des résultats ===
            st.subheader("🏆 Recommandations")
            
            successful_models = [r for r in results if '✅' in r['Status']]
            
            if successful_models:
                # Trouver le meilleur modèle par accuracy
                best_accuracy_model = max(successful_models, 
                                        key=lambda x: float(x['Accuracy']) if x['Accuracy'] != 'N/A' else 0)
                
                # Trouver le modèle le plus rapide
                fastest_model = min(successful_models,
                                  key=lambda x: float(x['Temps/pred (ms)']) if x['Temps/pred (ms)'] != 'N/A' else float('inf'))
                
                st.success(f"🎯 **Meilleure précision**: {best_accuracy_model['Modèle']} ({best_accuracy_model['Accuracy (%)']})")
                st.info(f"⚡ **Plus rapide**: {fastest_model['Modèle']} ({fastest_model['Temps/pred (ms)']} ms/pred)")
                
                # Recommandation globale
                st.subheader("💡 Recommandation générale")
                
                if best_accuracy_model['Modèle'] == fastest_model['Modèle']:
                    st.success(f"🏆 **Choix optimal**: {best_accuracy_model['Modèle']} (meilleur sur tous les critères)")
                else:
                    st.info("⚖️ **Compromis à faire**:")
                    st.write(f"- Pour la **précision**: {best_accuracy_model['Modèle']}")
                    st.write(f"- Pour la **vitesse**: {fastest_model['Modèle']}")
                
                # Informations sur Grad-CAM
                st.subheader("🧠 Compatibilité Grad-CAM")
                for result in successful_models:
                    if "Wrapper" in result['Modèle']:
                        st.warning(f"⚠️ {result['Modèle']}: Grad-CAM non disponible")
                    else:
                        st.success(f"✅ {result['Modèle']}: Grad-CAM disponible")
                
            else:
                st.error("❌ Aucun modèle n'a pu être évalué avec succès")
                st.info("💡 Vérifiez vos fichiers de modèles ou leurs formats")
        
        else:
            st.error("❌ Aucun résultat obtenu")

    # === Informations additionnelles ===
    st.markdown("---")
    st.subheader("ℹ️ À propos des tests")
    st.markdown("""
    **Métriques évaluées:**
    - **Accuracy**: Précision sur les données de test (synthétiques)
    - **Temps/prédiction**: Vitesse d'inférence en millisecondes
    - **Confiance moyenne**: Confiance moyenne du modèle dans ses prédictions
    - **Compatibilité Grad-CAM**: Disponibilité de la visualisation des zones d'attention
    
    **Note**: Les données de test sont synthétiques. Pour une évaluation réelle, 
    utilisez vos vraies données de validation.
    """)

if __name__ == "__main__":
    main()