import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# === Variables globales pour le cache ===
converted_model_cache = None

# === Fonction de conversion avec noms uniques ===
def convert_sequential_to_functional_unique_names(model):
    """
    Convertir un modèle Sequential en Functional avec des noms uniques
    """
    try:
        # Nettoyer la session pour éviter les conflits de noms
        tf.keras.backend.clear_session()
        
        # Créer l'input avec un nom unique
        input_layer = tf.keras.layers.Input(shape=(64, 64, 3), name='gradcam_input')
        
        # Passer l'input à travers toutes les couches
        x = input_layer
        
        for i, layer in enumerate(model.layers):
            # Créer une copie de la couche avec un nom unique
            layer_config = layer.get_config()
            layer_class = type(layer)
            
            # Assigner un nom unique si nécessaire
            if 'name' in layer_config:
                layer_config['name'] = f"gradcam_{layer_config['name']}_{i}"
            
            # Créer la nouvelle couche
            new_layer = layer_class.from_config(layer_config)
            
            # Copier les poids si ils existent
            if layer.weights:
                new_layer.build(x.shape)
                new_layer.set_weights(layer.get_weights())
            
            x = new_layer(x)
        
        # Créer le modèle Functional avec un nom unique
        functional_model = tf.keras.Model(
            inputs=input_layer, 
            outputs=x, 
            name='gradcam_functional_model'
        )
        
        return functional_model
        
    except Exception as e:
        st.error(f"❌ Erreur conversion: {e}")
        return None

# === Fonction Grad-CAM optimisée pour Streamlit ===
def make_gradcam_for_streamlit(img_array, model, layer_search_name, pred_index=None):
    """
    Version Grad-CAM optimisée pour Streamlit
    """
    global converted_model_cache
    
    try:
        # Utiliser le cache si disponible
        if converted_model_cache is None:
            if isinstance(model, tf.keras.Sequential):
                with st.spinner("🔄 Conversion du modèle (une seule fois)..."):
                    converted_model_cache = convert_sequential_to_functional_unique_names(model)
                    if converted_model_cache is None:
                        return None
                st.success("✅ Modèle converti avec succès")
            else:
                converted_model_cache = model
        
        working_model = converted_model_cache
        
        # Trouver la couche convolutionnelle
        conv_layer = None
        for layer in working_model.layers:
            if layer_search_name in layer.name:
                conv_layer = layer
                break
        
        if conv_layer is None:
            st.error(f"Couche contenant '{layer_search_name}' introuvable")
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

# === Diagnostic du modèle ===
def diagnose_model_streamlit(model):
    """Diagnostic du modèle pour Streamlit"""
    try:
        st.info("🔍 Analyse du modèle...")
        
        # Informations de base
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

# === Chargement robuste du modèle ===
@st.cache_resource
def load_skin_cancer_model_robust():
    try:
        st.info("🔄 Chargement du modèle...")
        
        # Nettoyer la session
        tf.keras.backend.clear_session()
        
        # Charger le modèle
        model = load_model("skin_cancer_model.h5", compile=False)
        st.success("✅ Modèle chargé avec succès")
        
        # Construire le modèle si nécessaire
        if not model.built:
            model.build(input_shape=(None, 64, 64, 3))
        
        # Test de fonctionnement
        dummy_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
        _ = model(dummy_input, training=False)
        st.success("✅ Test de prédiction réussi")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Erreur critique: {e}")
        raise e

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
    st.info("💡 Solutions possibles:")
    st.write("- Vérifiez que le fichier 'skin_cancer_model.h5' existe")
    st.write("- Le modèle pourrait être corrompu")
    st.write("- Essayez de réentraîner le modèle")
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
            # Utiliser le modèle converti s'il existe, sinon le modèle original
            if converted_model_cache is not None:
                prediction = converted_model_cache.predict(img_array, verbose=0)
            else:
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
            # Grad-CAM
            st.subheader("🧠 Analyse Grad-CAM")
            
            if conv_layers:
                # Utiliser la dernière couche convolutionnelle par défaut
                layer_to_use = 'conv2d_4'  # ou la dernière couche disponible
                
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
            else:
                st.warning("⚠️ Pas de couches convolutionnelles détectées")
                st.info("Le modèle ne semble pas compatible avec Grad-CAM")
                
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

