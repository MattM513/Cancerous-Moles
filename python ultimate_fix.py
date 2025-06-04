# === SOLUTION DÉFINITIVE POUR GRAD-CAM AVEC MODÈLES SEQUENTIAL ===
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convert_sequential_to_functional(model):
    """
    Convertir un modèle Sequential en modèle Functional pour résoudre le problème Grad-CAM
    """
    print("🔄 Conversion Sequential → Functional...")
    
    try:
        # Créer l'input explicite
        input_layer = tf.keras.layers.Input(shape=(64, 64, 3))
        
        # Passer l'input à travers toutes les couches du modèle Sequential
        x = input_layer
        for layer in model.layers:
            x = layer(x)
        
        # Créer le nouveau modèle Functional
        functional_model = tf.keras.Model(inputs=input_layer, outputs=x)
        
        print("✅ Conversion réussie")
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

def make_gradcam_ultimate(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Version ultime de Grad-CAM qui fonctionne avec tous les modèles
    """
    try:
        # S'assurer que c'est un modèle Functional
        if isinstance(model, tf.keras.Sequential):
            print("⚠️ Modèle Sequential détecté, conversion en cours...")
            model = convert_sequential_to_functional(model)
            if model is None:
                raise ValueError("Impossible de convertir le modèle")
        
        # Vérifier que la couche existe
        try:
            conv_layer = model.get_layer(last_conv_layer_name)
            print(f"✅ Couche trouvée: {conv_layer.name} ({type(conv_layer).__name__})")
        except ValueError:
            raise ValueError(f"Couche '{last_conv_layer_name}' introuvable")
        
        # S'assurer que l'entrée est un tensor
        if not isinstance(img_array, tf.Tensor):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        print(f"🎯 Input shape: {img_array.shape}")
        
        # Créer le modèle Grad-CAM
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )
        
        print("✅ Modèle Grad-CAM créé")
        
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
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"❌ Erreur Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_all_conv_layers_ultimate(model):
    """
    Tester toutes les couches convolutionnelles avec la méthode ultime
    """
    print("\n🧪 TEST ULTIME DE TOUTES LES COUCHES")
    print("=" * 50)
    
    # Convertir en Functional si nécessaire
    if isinstance(model, tf.keras.Sequential):
        functional_model = convert_sequential_to_functional(model)
        if functional_model is None:
            print("❌ Impossible de convertir le modèle")
            return []
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
    
    for layer_name in conv_layers:
        print(f"\n🔍 Test de {layer_name}...")
        
        heatmap = make_gradcam_ultimate(test_input, functional_model, layer_name)
        
        if heatmap is not None:
            working_layers.append(layer_name)
            print(f"✅ {layer_name}: SUCCÈS!")
            print(f"   Shape: {heatmap.shape}")
            print(f"   Min/Max: {heatmap.min():.3f}/{heatmap.max():.3f}")
        else:
            print(f"❌ {layer_name}: ÉCHEC")
    
    print(f"\n🎉 RÉSULTATS FINAUX:")
    if working_layers:
        print(f"✅ Couches compatibles: {working_layers}")
        print(f"💡 Recommandée: '{working_layers[-1]}'")
    else:
        print("❌ Aucune couche compatible")
    
    return working_layers, functional_model

def create_gradcam_visualization(img_array, model, layer_name, class_names=None):
    """
    Créer une visualisation complète avec Grad-CAM
    """
    try:
        print(f"\n🎨 Création visualisation Grad-CAM pour {layer_name}")
        
        # Faire la prédiction
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        print(f"🎯 Prédiction: Classe {predicted_class} ({confidence:.1f}%)")
        
        # Générer la heatmap
        heatmap = make_gradcam_ultimate(img_array, model, layer_name, predicted_class)
        
        if heatmap is None:
            return None
        
        # Redimensionner la heatmap à la taille de l'image originale
        img_size = img_array.shape[1:3]  # (height, width)
        heatmap_resized = cv2.resize(heatmap, (img_size[1], img_size[0]))
        
        # Convertir en couleur
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        # Image originale
        img_original = np.uint8(255 * img_array[0])
        
        # Superposition
        superimposed = cv2.addWeighted(img_original, 0.6, heatmap_colored, 0.4, 0)
        
        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image originale
        axes[0].imshow(img_original)
        axes[0].set_title("Image originale")
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title(f"Grad-CAM ({layer_name})")
        axes[1].axis('off')
        
        # Superposition
        axes[2].imshow(superimposed)
        axes[2].set_title(f"Superposition\nClasse: {predicted_class} ({confidence:.1f}%)")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        return fig, heatmap_resized, superimposed
        
    except Exception as e:
        print(f"❌ Erreur visualisation: {e}")
        return None

# === FONCTION PRINCIPALE DE TEST ===
def main_ultimate_test():
    """
    Test complet de la solution ultime
    """
    print("🚀 TEST ULTIME DE LA SOLUTION GRAD-CAM")
    print("=" * 60)
    
    try:
        # Charger le modèle
        print("📁 Chargement du modèle...")
        model = tf.keras.models.load_model("skin_cancer_model.h5", compile=False)
        print(f"✅ Modèle chargé: {type(model).__name__}")
        
        # Tester toutes les couches
        working_layers, functional_model = test_all_conv_layers_ultimate(model)
        
        if working_layers:
            print(f"\n🎉 SUCCÈS! La solution fonctionne!")
            
            # Test avec une vraie image
            print(f"\n🖼️ Test avec image de démonstration...")
            test_img = np.random.rand(1, 64, 64, 3).astype(np.float32)  # Image aléatoire pour test
            
            # Utiliser la dernière couche convolutionnelle
            best_layer = working_layers[-1]
            
            result = create_gradcam_visualization(test_img, functional_model, best_layer)
            
            if result:
                fig, heatmap, superimposed = result
                plt.savefig("gradcam_test_result.png", dpi=150, bbox_inches='tight')
                print("✅ Visualisation sauvegardée: gradcam_test_result.png")
                plt.show()
            
            # Sauvegarder le modèle fonctionnel
            functional_model.save("skin_cancer_model_functional.h5")
            print("💾 Modèle fonctionnel sauvegardé: skin_cancer_model_functional.h5")
            
            return functional_model, working_layers
        else:
            print(f"\n❌ Aucune solution trouvée")
            return None, []
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()
        return None, []

if __name__ == "__main__":
    model, layers = main_ultimate_test()
    
    if model and layers:
        print(f"\n" + "="*60)  
        print("🎯 INSTRUCTIONS POUR VOTRE APP STREAMLIT:")
        print("="*60)
        print(f"1. Utilisez le fichier: 'skin_cancer_model_functional.h5'")
        print(f"2. Couche Grad-CAM recommandée: '{layers[-1]}'")
        print(f"3. Utilisez la fonction 'make_gradcam_ultimate()' dans votre code")
        print(f"4. Le modèle est maintenant compatible avec Grad-CAM!")