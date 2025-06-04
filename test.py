# === Solution pour fixer le problème Grad-CAM ===
import tensorflow as tf
import numpy as np

def fix_sequential_model_for_gradcam(model):
    """
    Corrige un modèle Sequential pour le rendre compatible avec Grad-CAM
    """
    # Vérifier si le modèle est déjà construit
    if not model.built:
        # Construire le modèle avec la forme d'entrée correcte
        model.build(input_shape=(None, 64, 64, 3))
    
    # Faire un appel initial pour initialiser les couches
    dummy_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
    _ = model(dummy_input, training=False)
    
    return model

def make_gradcam_heatmap_fixed(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Version corrigée de Grad-CAM pour les modèles Sequential
    """
    try:
        # S'assurer que le modèle est correctement initialisé
        model = fix_sequential_model_for_gradcam(model)
        
        # Vérifier que la couche existe
        try:
            conv_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            raise ValueError(f"Couche '{last_conv_layer_name}' introuvable")
        
        # S'assurer que l'entrée est un tensor
        if not isinstance(img_array, tf.Tensor):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Créer le modèle Grad-CAM - SOLUTION CLÉE
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
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
            raise ValueError("Gradients None - couche non différentiable")
        
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
        raise Exception(f"Erreur Grad-CAM: {str(e)}")

def test_layer_gradcam_compatibility_fixed(model, layer_name, test_input):
    """
    Test corrigé de compatibilité Grad-CAM
    """
    try:
        # Fixer le modèle d'abord
        model = fix_sequential_model_for_gradcam(model)
        
        # Tester Grad-CAM
        make_gradcam_heatmap_fixed(test_input, model, layer_name)
        return True, "OK"
    except Exception as e:
        return False, str(e)

# === Test rapide ===
def quick_test_gradcam_fix():
    """
    Test rapide pour vérifier si la correction fonctionne
    """
    try:
        # Charger le modèle
        model = tf.keras.models.load_model("skin_cancer_model.h5", compile=False)
        
        # Appliquer la correction
        model = fix_sequential_model_for_gradcam(model)
        
        # Créer une image de test
        test_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
        
        # Tester chaque couche convolutionnelle
        conv_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']
        
        working_layers = []
        for layer_name in conv_layers:
            is_compatible, error_msg = test_layer_gradcam_compatibility_fixed(model, layer_name, test_input)
            
            if is_compatible:
                working_layers.append(layer_name)
                print(f"✅ {layer_name}: Compatible")
            else:
                print(f"❌ {layer_name}: {error_msg}")
        
        if working_layers:
            print(f"\n🎯 Couches compatibles: {working_layers}")
            print(f"💡 Utilisez '{working_layers[-1]}' pour Grad-CAM")
        else:
            print("\n❌ Aucune couche compatible trouvée")
        
        return working_layers
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return []

if __name__ == "__main__":
    print("🔧 Test de la correction Grad-CAM...")
    working_layers = quick_test_gradcam_fix()