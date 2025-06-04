# === SOLUTION FINALE POUR LE PROBLÈME "input_layer" UTILISÉ 2 FOIS ===
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

def convert_sequential_to_functional_unique_names(model):
    """
    Convertir un modèle Sequential en Functional avec des noms uniques
    """
    print("🔄 Conversion Sequential → Functional (noms uniques)...")
    
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
        
        print("✅ Conversion réussie avec noms uniques")
        print(f"   - Type: {type(functional_model).__name__}")
        print(f"   - Input: {functional_model.input.shape}")
        print(f"   - Output: {functional_model.output.shape}")
        
        # Test de fonctionnement
        dummy_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
        output = functional_model(dummy_input, training=False)
        print(f"   - Test OK: {output.shape}")
        
        return functional_model
        
    except Exception as e:
        print(f"❌ Erreur conversion: {e}")
        return None

def make_gradcam_final(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Version finale de Grad-CAM qui résout le problème des noms dupliqués
    """
    try:
        # Conversion en Functional avec noms uniques si nécessaire
        if isinstance(model, tf.keras.Sequential):
            print("⚠️ Modèle Sequential détecté, conversion en cours...")
            model = convert_sequential_to_functional_unique_names(model)
            if model is None:
                raise ValueError("Impossible de convertir le modèle")
        
        # Trouver la couche convolutionnelle
        conv_layer = None
        for layer in model.layers:
            if last_conv_layer_name in layer.name:
                conv_layer = layer
                break
        
        if conv_layer is None:
            raise ValueError(f"Couche contenant '{last_conv_layer_name}' introuvable")
        
        print(f"✅ Couche trouvée: {conv_layer.name} ({type(conv_layer).__name__})")
        
        # S'assurer que l'entrée est un tensor
        if not isinstance(img_array, tf.Tensor):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        print(f"🎯 Input shape: {img_array.shape}")
        
        # Créer le modèle Grad-CAM avec un nom unique
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output],
            name='unique_gradcam_model'
        )
        
        print("✅ Modèle Grad-CAM créé avec succès")
        
        # Calcul des gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = grad_model(img_array, training=False)
            
            print(f"🔥 Conv outputs: {conv_outputs.shape}")
            print(f"📊 Predictions: {predictions.shape}")
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
                print(f"🎯 Classe prédite: {pred_index}")
            
            class_channel = predictions[:, pred_index]
        
        # Obtenir les gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            raise ValueError("Gradients None - couche non différentiable")
        
        print(f"⚡ Gradients: {grads.shape}")
        
        # Calculer la carte de chaleur
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        print(f"🔥 Heatmap brute: {heatmap.shape}")
        
        # Normalisation
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        
        if max_val > 1e-8:
            heatmap = heatmap / max_val
        else:
            heatmap = tf.ones_like(heatmap) * 0.5
        
        print(f"✅ Heatmap finale: {heatmap.shape}")
        
        return heatmap.numpy(), model  # Retourner aussi le modèle converti
        
    except Exception as e:
        print(f"❌ Erreur Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_all_layers_final(model):
    """
    Test final de toutes les couches avec la solution des noms uniques
    """
    print("\n🧪 TEST FINAL AVEC NOMS UNIQUES")
    print("=" * 50)
    
    # Convertir en Functional si nécessaire
    if isinstance(model, tf.keras.Sequential):
        functional_model = convert_sequential_to_functional_unique_names(model)
        if functional_model is None:
            print("❌ Impossible de convertir le modèle")
            return [], None
    else:
        functional_model = model
    
    # Trouver les couches convolutionnelles
    conv_layers = []
    for layer in functional_model.layers:
        if hasattr(layer, 'filters') and hasattr(layer, 'kernel_size'):
            conv_layers.append(layer.name)
    
    print(f"🎯 Couches convolutionnelles: {conv_layers}")
    
    # Créer une image de test
    test_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
    
    working_layers = []
    final_model = functional_model
    
    for layer_name in conv_layers:
        print(f"\n🔍 Test de {layer_name}...")
        
        # Utiliser une version simplifiée du nom pour la recherche
        simple_name = layer_name.replace('gradcam_', '').split('_')[0]
        
        heatmap, converted_model = make_gradcam_final(test_input, functional_model, simple_name)
        
        if heatmap is not None:
            working_layers.append((layer_name, simple_name))
            final_model = converted_model
            print(f"✅ {layer_name}: SUCCÈS!")
            print(f"   Shape: {heatmap.shape}")
            print(f"   Min/Max: {heatmap.min():.3f}/{heatmap.max():.3f}")
        else:
            print(f"❌ {layer_name}: ÉCHEC")
    
    print(f"\n🎉 RÉSULTATS FINAUX:")
    if working_layers:
        print(f"✅ Couches compatibles: {[name for name, _ in working_layers]}")
        print(f"💡 Recommandée: '{working_layers[-1][1]}'")
    else:
        print("❌ Aucune couche compatible")
    
    return working_layers, final_model

# === FONCTION STREAMLIT MISE À JOUR ===
def make_gradcam_for_streamlit(img_array, model, layer_search_name, pred_index=None):
    """
    Version Grad-CAM optimisée pour Streamlit
    """
    global converted_model_cache
    
    try:
        # Utiliser le cache si disponible
        if 'converted_model_cache' not in globals():
            if isinstance(model, tf.keras.Sequential):
                st.info("🔄 Conversion du modèle en cours (une seule fois)...")
                converted_model_cache = convert_sequential_to_functional_unique_names(model)
                if converted_model_cache is None:
                    return None
                st.success("✅ Modèle converti avec succès")
            else:
                converted_model_cache = model
        
        model = converted_model_cache
        
        # Trouver la couche convolutionnelle
        conv_layer = None
        for layer in model.layers:
            if layer_search_name in layer.name:
                conv_layer = layer
                break
        
        if conv_layer is None:
            st.error(f"Couche contenant '{layer_search_name}' introuvable")
            return None
        
        # S'assurer que l'entrée est un tensor
        if not isinstance(img_array, tf.Tensor):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Créer le modèle Grad-CAM
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output],
            name=f'gradcam_{layer_search_name}_{np.random.randint(1000)}'
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
        return None

# === TEST PRINCIPAL ===
def main_final_test():
    """
    Test principal de la solution finale
    """
    print("🚀 TEST FINAL DE LA SOLUTION GRAD-CAM")
    print("=" * 60)
    
    try:
        # Charger le modèle
        print("📁 Chargement du modèle...")
        model = tf.keras.models.load_model("skin_cancer_model.h5", compile=False)
        print(f"✅ Modèle chargé: {type(model).__name__}")
        
        # Tester toutes les couches
        working_layers, final_model = test_all_layers_final(model)
        
        if working_layers:
            print(f"\n🎉 SUCCÈS! La solution fonctionne!")
            
            # Sauvegarder le modèle converti
            final_model.save("skin_cancer_model_gradcam_ready.h5")
            print("💾 Modèle optimisé sauvegardé: skin_cancer_model_gradcam_ready.h5")
            
            # Instructions finales
            print(f"\n" + "="*60)  
            print("🎯 INSTRUCTIONS POUR VOTRE APP STREAMLIT:")
            print("="*60)
            print(f"1. Utilisez: 'skin_cancer_model_gradcam_ready.h5'")
            print(f"2. Couche Grad-CAM: '{working_layers[-1][1]}'")
            print(f"3. Utilisez la fonction 'make_gradcam_for_streamlit()'")
            print(f"4. Le problème des noms dupliqués est résolu!")
            
            return final_model, working_layers
        else:
            print(f"\n❌ Aucune solution trouvée")
            return None, []
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()
        return None, []

if __name__ == "__main__":
    model, layers = main_final_test()