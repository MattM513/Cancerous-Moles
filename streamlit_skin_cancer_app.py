import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# === Variables globales pour le cache ===
converted_model_cache = None

# === Diagnostic des fichiers ===
files_to_check = [
    "skin_cancer_model.weights.h5",
    "skin_cancer_model.h5", 
    "skin_cancer_model.keras",
    "skin_cancer_model_savedmodel"
]

st.write("🔍 **Fichiers de modèle disponibles:**")
for file in files_to_check:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        st.write(f"✅ {file} trouvé ({size_mb:.1f} MB)")
    else:
        st.write(f"❌ {file} non trouvé")

# === Fonction Grad-CAM optimisée pour Streamlit ===
def make_gradcam_for_streamlit(img_array, model, layer_search_name, pred_index=None):
    """
    Version Grad-CAM simplifiée pour les modèles déjà fonctionnels
    """
    try:
        # SIMPLIFICATION MAJEURE : Plus besoin de conversion ou de cache complexe
        working_model = model
        
        # Trouver la couche convolutionnelle
        conv_layer = None
        for layer in working_model.layers:
            # On cherche un nom exact car les noms peuvent être complexes
            if layer.name == layer_search_name:
                conv_layer = layer
                break
        
        if conv_layer is None:
            st.error(f"Couche '{layer_search_name}' introuvable. Couches disponibles: {[l.name for l in working_model.layers]}")
            return None
        
        # S'assurer que l'entrée est un tensor
        if not isinstance(img_array, tf.Tensor):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Créer le modèle Grad-CAM avec un nom unique basé sur le timestamp
        import time
        unique_id = str(int(time.time() * 1000))[-6:]
        
        grad_model = tf.keras.models.Model(
            inputs=working_model.input,
            outputs=[conv_layer.output, working_model.output],
            name=f'gradcam_model_{unique_id}'
        )
        
        # Calcul des gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = grad_model(img_array, training=False)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Obtenir les gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            st.error("Gradients None - couche non différentiable")
            return None
        
        # Calculer la carte de chaleur
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalisation
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        
        if max_val > 1e-8:
            heatmap = heatmap / max_val
        else:
            heatmap = tf.ones_like(heatmap) * 0.5
        
        return heatmap.numpy()
        
    except Exception as e:
        st.error(f"❌ Erreur Grad-CAM: {e}")
        with st.expander("🔍 Détails de l'erreur"):
            import traceback
            st.code(traceback.format_exc())
        return None

# === Diagnostic du modèle adapté ===
def diagnose_model_streamlit(model):
    """Diagnostic du modèle pour Streamlit (compatible avec wrapper)"""
    try:
        st.info("🔍 Analyse du modèle...")
        
        # Vérifier si c'est notre wrapper ou un modèle Keras normal
        if isinstance(model, SavedModelWrapper):
            st.info("📋 Modèle: SavedModel avec wrapper")
            st.warning("⚠️ Grad-CAM non disponible avec SavedModel")
            return []  # Pas de couches conv disponibles pour Grad-CAM
        
        # Modèle Keras normal
        model_type = type(model).__name__
        layer_count = len(model.layers)
        
        # Trouver les couches convolutionnelles
        conv_layers = []
        all_layers_info = []
        
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            layer_name = layer.name
            all_layers_info.append(f"{layer_name} ({layer_type})")
            
            # Vérifier si c'est une vraie couche convolutionnelle
            if hasattr(layer, 'filters') and hasattr(layer, 'kernel_size'):
                conv_layers.append(layer.name)
        
        # Affichage des informations
        st.success(f"📊 Modèle: {model_type} avec {layer_count} couches")
        
        if conv_layers:
            st.success(f"🎯 Couches convolutionnelles: {conv_layers}")
        else:
            st.warning("⚠️ Aucune couche convolutionnelle détectée")
        
        return conv_layers
        
    except Exception as e:
        st.error(f"❌ Erreur lors du diagnostic: {e}")
        return []

# === WRAPPER POUR SAVEDMODEL ===
class SavedModelWrapper:
    """Wrapper pour utiliser un SavedModel comme un modèle Keras"""
    
    def __init__(self, savedmodel_path):
        self.model = tf.saved_model.load(savedmodel_path)
        self.serving_fn = self.model.signatures['serving_default']
        
        # Analyser la signature pour comprendre les entrées/sorties
        self.input_key = list(self.serving_fn.structured_input_signature[1].keys())[0]
        self.output_key = list(self.serving_fn.structured_outputs.keys())[0]
        
        st.info(f"📋 Input key: {self.input_key}")
        st.info(f"📋 Output key: {self.output_key}")
    
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
    
    @property
    def layers(self):
        """Propriété layers fictive pour compatibilité"""
        return []

# === NOUVELLE FONCTION DE CHARGEMENT CORRIGÉE ===
@st.cache_resource
def load_skin_cancer_model_robust():
    """
    Charge le modèle en essayant plusieurs méthodes dans l'ordre de préférence
    """
    
    # MÉTHODE 1: SavedModel avec wrapper (PRIORITÉ car c'est ce qui fonctionne)
    if os.path.isdir("skin_cancer_model_savedmodel"):
        try:
            # Vérifier que le dossier contient bien des fichiers
            savedmodel_files = []
            for root, dirs, files in os.walk("skin_cancer_model_savedmodel"):
                savedmodel_files.extend(files)
            
            if len(savedmodel_files) > 0:
                st.info(f"🔄 Chargement du SavedModel... ({len(savedmodel_files)} fichiers trouvés)")
                
                # D'abord, essayer le chargement Keras normal
                try:
                    model = tf.keras.models.load_model("skin_cancer_model_savedmodel")
                    if hasattr(model, 'predict') and hasattr(model, 'layers'):
                        st.success("✅ SavedModel Keras chargé avec succès !")
                        return model
                except:
                    pass
                
                # Si ça ne marche pas, utiliser notre wrapper
                st.info("🔄 Utilisation du wrapper SavedModel...")
                wrapped_model = SavedModelWrapper("skin_cancer_model_savedmodel")
                st.success("✅ SavedModel avec wrapper chargé avec succès !")
                return wrapped_model
            else:
                st.warning("⚠️ Dossier SavedModel vide")
                
        except Exception as e:
            st.warning(f"❌ Échec SavedModel: {str(e)[:100]}...")
            with st.expander("🔍 Détails erreur SavedModel"):
                st.code(str(e))
    
    # MÉTHODE 2: Essayer .h5 avec des options compatibilité
    if os.path.exists("skin_cancer_model.h5"):
        try:
            st.info("🔄 Chargement du modèle .h5 (mode compatibilité)...")
            model = tf.keras.models.load_model("skin_cancer_model.h5", compile=False)
            st.success("✅ Modèle .h5 chargé avec succès !")
            return model
        except Exception as e:
            st.warning(f"❌ Échec .h5: {str(e)[:100]}...")
    
    # MÉTHODE 3: Essayer .keras avec des options compatibilité
    if os.path.exists("skin_cancer_model.keras"):
        try:
            st.info("🔄 Chargement du modèle .keras (mode compatibilité)...")
            model = tf.keras.models.load_model("skin_cancer_model.keras", compile=False)
            st.success("✅ Modèle .keras chargé avec succès !")
            return model
        except Exception as e:
            st.warning(f"❌ Échec .keras: {str(e)[:100]}...")
    
    # Si aucune méthode n'a fonctionné
    st.error("❌ Toutes les méthodes de chargement ont échoué")
    st.info("💡 **Solutions:**")
    st.write("1. Le SavedModel semble être votre meilleur option")
    st.write("2. Vérifiez votre version de TensorFlow")
    st.write("3. Essayez: `pip install tensorflow==2.15.0`")
    st.write("4. Ou régénérez le modèle avec votre version actuelle")
    
    raise Exception("Impossible de charger le modèle avec toutes les méthodes disponibles")

# === Noms et descriptions des classes ===
class_names = {
    'akiec': "Carcinome intraépithélial actinique",
    'bcc': "Carcinome basocellulaire", 
    'bkl': "Kératose bénigne",
    'df': "Dermatofibrome",
    'mel': "Mélanome",
    'nv': "Grain de beauté bénin",
    'vasc': "Lésion vasculaire"
}

class_descriptions = {
    'akiec': "Lésion précancéreuse (maligne)",
    'bcc': "Cancer de la peau à croissance lente",
    'bkl': "Lésion non cancéreuse de la peau", 
    'df': "Tumeur cutanée bénigne rare",
    'mel': "Cancer de la peau dangereux",
    'nv': "Grain de beauté fréquent, non cancéreux",
    'vasc': "Lésion des vaisseaux sanguins bénigne"
}

# === Interface Streamlit ===
st.title("🌿 Détection de grains de beauté cancérigènes")
st.markdown("Uploadez une image et le modèle prédit sa classe avec visualisation Grad-CAM.")

# === Chargement du modèle ===
try:
    model = load_skin_cancer_model_robust()
    
    # Diagnostic du modèle
    conv_layers = diagnose_model_streamlit(model)
    
except Exception as e:
    st.error(f"❌ Impossible de charger le modèle: {e}")
    st.info("💡 **Solutions possibles:**")
    st.write("- Vérifiez que vos fichiers de modèle (.h5, .keras) ne sont pas corrompus")
    st.write("- Réentraînez votre modèle si nécessaire")
    st.write("- Assurez-vous d'avoir sauvegardé le modèle complet (pas seulement les poids)")
    st.stop()

# === Interface utilisateur ===
uploaded_file = st.file_uploader("Choisissez une image JPG ou PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Préprocessing de l'image
        img = Image.open(uploaded_file).convert('RGB').resize((64, 64))
        st.image(img, caption="Image chargée", use_column_width=False)
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction
        with st.spinner("🔮 Analyse en cours..."):
            prediction = model.predict(img_array, verbose=0)
            
            predicted_index = np.argmax(prediction)
            predicted_key = list(class_names.keys())[predicted_index]
            predicted_label = class_names[predicted_key]
            predicted_confidence = prediction[0][predicted_index] * 100
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        
        with col1:
            # Résultats de prédiction
            if prediction.max() < 0.4:
                st.warning("❌ Image très différente des exemples d'entraînement")
            else:
                st.success(f"✅ Classe prédite: **{predicted_label}**")
                st.info(f"ℹ️ {class_descriptions[predicted_key]}")
            
            st.info(f"🎯 Confiance: {predicted_confidence:.2f}%")
            
            if predicted_key in ['mel', 'akiec', 'bcc']:
                st.error("⚠️ Type potentiellement **dangereux** - Consultez un dermatologue")
            
            # Graphique camembert
            st.subheader("📊 Distribution des probabilités")
            labels = list(class_names.values())
            sizes = [prediction[0][i] * 100 for i in range(len(labels))]
            colors = ["#ffe6f0", "#ffcce0", "#ffb3d1", "#ff99c2", "#ff80b3", "#ff66a3", "#ff4d94"]
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                   colors=colors, textprops={'fontsize': 9})
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            st.subheader("🧠 Analyse Grad-CAM")
            if conv_layers:
                layer_to_use = conv_layers[-1] 
                
                st.info(f"🧠 Utilisation de la couche : **{layer_to_use}**")
                
                # Générer la heatmap
                with st.spinner("🔥 Génération de la carte de chaleur..."):
                    test_input = tf.convert_to_tensor(img_array, dtype=tf.float32)
                    heatmap = make_gradcam_for_streamlit(test_input, model, layer_to_use, predicted_index)
                
                if heatmap is not None:
                    try:
                        # Superposer sur l'image originale
                        img_rgb = np.array(img)
                        heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
                        heatmap_colored = np.uint8(255 * heatmap_resized)
                        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
                        
                        # Fusion des images
                        superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
                        
                        st.image(superimposed_img, 
                                caption=f"🎯 Zones d'attention du modèle", 
                                use_column_width=True)
                        
                        st.success("🔥 Rouge = zones importantes pour la décision")
                        st.info(f"📱 Couche utilisée: {layer_to_use}")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la visualisation: {e}")
                else:
                    st.error("❌ Impossible de générer la carte Grad-CAM")
            elif isinstance(model, SavedModelWrapper):
                st.info("ℹ️ **Grad-CAM non disponible avec SavedModel**")
                st.write("Le modèle fonctionne mais les couches internes ne sont pas accessibles.")
                st.write("Pour avoir Grad-CAM, utilisez un modèle .h5 ou .keras.")
                
                # Afficher quand même l'image originale
                st.image(img, caption="🖼️ Image analysée", use_column_width=True)
            else:
                st.warning("⚠️ Pas de couches convolutionnelles détectées")
                st.info("Le modèle ne semble pas compatible avec Grad-CAM")
                
                # Afficher quand même l'image originale
                st.image(img, caption="🖼️ Image analysée", use_column_width=True)
                
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement: {e}")
        with st.expander("🔍 Détails de l'erreur"):
            import traceback
            st.code(traceback.format_exc())

# === Informations ===
st.markdown("---")
st.subheader("ℹ️ À propos")
st.markdown("""
Cette application utilise un modèle de deep learning pour classifier les lésions cutanées:

**Classes détectées:**
- **Mélanome (mel)** ⚠️ : Cancer dangereux
- **Carcinome basocellulaire (bcc)** ⚠️ : Cancer commun 
- **Carcinome actinique (akiec)** ⚠️ : Lésion précancéreuse
- **Grain de beauté bénin (nv)** ✅ : Non cancéreux
- **Kératose bénigne (bkl)** ✅ : Lésion bénigne
- **Dermatofibrome (df)** ✅ : Tumeur bénigne
- **Lésion vasculaire (vasc)** ✅ : Anomalie bénigne

**Grad-CAM** montre les zones sur lesquelles le modèle se concentre pour prendre sa décision.
""")

st.error("⚠️ **AVERTISSEMENT MÉDICAL**: Application à but éducatif uniquement. Consultez un professionnel pour un diagnostic médical.")