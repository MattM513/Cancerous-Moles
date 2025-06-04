# === SOLUTION COMPLÈTE POUR LE PROBLÈME DU MODÈLE .H5 ===
import tensorflow as tf
import numpy as np
import os

def diagnose_h5_model(model_path="skin_cancer_model.h5"):
    """
    Diagnostic complet du modèle .h5 pour identifier le problème
    """
    print("🔍 DIAGNOSTIC DU MODÈLE .H5")
    print("=" * 50)
    
    # Vérifier l'existence du fichier
    if not os.path.exists(model_path):
        print(f"❌ Fichier {model_path} introuvable !")
        return None
    
    print(f"✅ Fichier {model_path} trouvé")
    
    try:
        # Charger le modèle
        print("\n🔄 Chargement du modèle...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Modèle chargé: {type(model).__name__}")
        
        # Vérifier l'état du modèle
        print(f"\n📊 État du modèle:")
        print(f"   - Built: {model.built}")
        print(f"   - Nombre de couches: {len(model.layers)}")
        
        # Analyser les couches
        print(f"\n🏗️ Structure des couches:")
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            layer_name = layer.name
            
            # Vérifier si la couche a une forme d'entrée définie
            try:
                input_shape = layer.input_shape if hasattr(layer, 'input_shape') else "Non définie"
                output_shape = layer.output_shape if hasattr(layer, 'output_shape') else "Non définie"
                print(f"   {i}: {layer_name} ({layer_type})")
                print(f"      Input: {input_shape}")
                print(f"      Output: {output_shape}")
            except Exception as e:
                print(f"   {i}: {layer_name} ({layer_type}) - Erreur: {e}")
        
        # Vérifier l'input du modèle
        print(f"\n🎯 Input du modèle:")
        try:
            if hasattr(model, 'input_shape'):
                print(f"   Input shape: {model.input_shape}")
            if hasattr(model, 'input'):
                print(f"   Input défini: {model.input is not None}")
        except Exception as e:
            print(f"   ❌ Erreur input: {e}")
        
        return model
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

def fix_h5_model_completely(model_path="skin_cancer_model.h5"):
    """
    Solution complète pour réparer le modèle .h5
    """
    print("\n🔧 RÉPARATION DU MODÈLE")
    print("=" * 50)
    
    # Diagnostic initial
    model = diagnose_h5_model(model_path)
    if model is None:
        return None
    
    try:
        # SOLUTION 1: Forcer la construction du modèle
        print("\n🔨 Tentative 1: Construction forcée...")
        
        if not model.built:
            print("   - Modèle non construit, construction en cours...")
            model.build(input_shape=(None, 64, 64, 3))
            print("   ✅ Modèle construit")
        
        # SOLUTION 2: Appel d'initialisation
        print("\n🔨 Tentative 2: Initialisation par appel...")
        dummy_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
        
        try:
            output = model(dummy_input, training=False)
            print(f"   ✅ Appel réussi - Output shape: {output.shape}")
        except Exception as e:
            print(f"   ⚠️ Erreur lors de l'appel: {e}")
            
            # SOLUTION 3: Reconstruction du modèle
            print("\n🔨 Tentative 3: Reconstruction complète...")
            model = reconstruct_model_from_h5(model)
        
        # Test final
        print("\n✅ TEST FINAL:")
        test_gradcam_compatibility(model)
        
        return model
        
    except Exception as e:
        print(f"❌ Erreur lors de la réparation: {e}")
        return None

def reconstruct_model_from_h5(original_model):
    """
    Reconstruire complètement le modèle à partir de ses poids
    """
    print("   🏗️ Reconstruction du modèle...")
    
    try:
        # Créer un nouveau modèle avec la même architecture
        new_model = tf.keras.Sequential()
        
        # Ajouter une couche d'entrée explicite
        new_model.add(tf.keras.layers.Input(shape=(64, 64, 3)))
        
        # Copier les couches (sauf Input si elle existe déjà)
        for layer in original_model.layers:
            if not isinstance(layer, tf.keras.layers.InputLayer):
                # Créer une nouvelle couche avec la même configuration
                layer_config = layer.get_config()
                layer_class = type(layer)
                new_layer = layer_class.from_config(layer_config)
                new_model.add(new_layer)
        
        # Construire le nouveau modèle
        new_model.build(input_shape=(None, 64, 64, 3))
        
        # Copier les poids
        print("   📋 Copie des poids...")
        for new_layer, old_layer in zip(new_model.layers[1:], original_model.layers):
            if old_layer.weights:
                new_layer.set_weights(old_layer.get_weights())
        
        # Test du nouveau modèle
        dummy_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
        output = new_model(dummy_input, training=False)
        print(f"   ✅ Nouveau modèle fonctionnel - Output: {output.shape}")
        
        return new_model
        
    except Exception as e:
        print(f"   ❌ Erreur reconstruction: {e}")
        return original_model

def test_gradcam_compatibility(model):
    """
    Tester la compatibilité Grad-CAM après réparation
    """
    print("\n🧪 Test compatibilité Grad-CAM:")
    
    # Trouver les couches convolutionnelles
    conv_layers = []
    for layer in model.layers:
        if 'conv' in layer.name.lower() and hasattr(layer, 'filters'):
            conv_layers.append(layer.name)
    
    if not conv_layers:
        print("   ❌ Aucune couche convolutionnelle trouvée")
        return []
    
    print(f"   🎯 Couches conv trouvées: {conv_layers}")
    
    # Tester chaque couche
    test_input = tf.zeros((1, 64, 64, 3), dtype=tf.float32)
    working_layers = []
    
    for layer_name in conv_layers:
        try:
            # Test simple de Grad-CAM
            conv_layer = model.get_layer(layer_name)
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[conv_layer.output, model.output]
            )
            
            with tf.GradientTape() as tape:
                tape.watch(test_input)
                conv_outputs, predictions = grad_model(test_input, training=False)
                class_channel = predictions[:, 0]
            
            grads = tape.gradient(class_channel, conv_outputs)
            
            if grads is not None:
                working_layers.append(layer_name)
                print(f"   ✅ {layer_name}: Compatible")
            else:
                print(f"   ❌ {layer_name}: Gradients None")
                
        except Exception as e:
            print(f"   ❌ {layer_name}: {str(e)[:50]}...")
    
    if working_layers:
        print(f"\n🎉 SUCCÈS! Couches compatibles: {working_layers}")
        print(f"💡 Utilisez '{working_layers[-1]}' pour Grad-CAM")
    else:
        print("\n❌ Aucune couche compatible")
    
    return working_layers

def save_fixed_model(model, output_path="skin_cancer_model_fixed.h5"):
    """
    Sauvegarder le modèle réparé
    """
    try:
        print(f"\n💾 Sauvegarde du modèle réparé: {output_path}")
        model.save(output_path)
        print("✅ Modèle sauvegardé avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return False

# === FONCTION PRINCIPALE ===
def fix_main():
    """
    Fonction principale pour réparer le modèle
    """
    print("🚀 RÉPARATION COMPLÈTE DU MODÈLE .H5")
    print("=" * 60)
    
    # Étape 1: Diagnostic
    model = diagnose_h5_model()
    if model is None:
        print("\n❌ Impossible de diagnostiquer le modèle")
        return None
    
    # Étape 2: Réparation
    fixed_model = fix_h5_model_completely()
    if fixed_model is None:
        print("\n❌ Impossible de réparer le modèle")
        return None
    
    # Étape 3: Test final
    working_layers = test_gradcam_compatibility(fixed_model)
    
    # Étape 4: Sauvegarde
    if working_layers:
        save_fixed_model(fixed_model)
        print(f"\n🎉 RÉPARATION TERMINÉE AVEC SUCCÈS!")
        print(f"📁 Utilisez 'skin_cancer_model_fixed.h5'")
        print(f"🎯 Couche Grad-CAM recommandée: '{working_layers[-1]}'")
    else:
        print(f"\n⚠️ Modèle réparé mais Grad-CAM non fonctionnel")
    
    return fixed_model

# === ALTERNATIVE: RECRÉER LE MODÈLE ===
def create_new_model_architecture():
    """
    Si la réparation échoue, créer un nouveau modèle avec une architecture similaire
    """
    print("\n🏗️ CRÉATION D'UN NOUVEAU MODÈLE")
    print("=" * 50)
    
    try:
        # Architecture typique pour classification d'images médicales
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            
            # Premier bloc convolutionnel
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Deuxième bloc
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Troisième bloc
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_2'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Quatrième bloc
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv2d_3'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Cinquième bloc
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv2d_4'),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Classification
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 classes
        ])
        
        print("✅ Nouveau modèle créé")
        print(f"📊 Couches: {len(model.layers)}")
        
        # Test de compatibilité
        working_layers = test_gradcam_compatibility(model)
        
        if working_layers:
            print("✅ Nouveau modèle compatible avec Grad-CAM")
            model.save("skin_cancer_model_new_architecture.h5")
            print("💾 Modèle sauvegardé: skin_cancer_model_new_architecture.h5")
            print("⚠️ ATTENTION: Ce modèle n'est pas entraîné!")
            print("🔄 Vous devrez le réentraîner avec vos données")
        
        return model
        
    except Exception as e:
        print(f"❌ Erreur création nouveau modèle: {e}")
        return None

if __name__ == "__main__":
    # Essayer de réparer le modèle existant
    fixed_model = fix_main()
    
    if fixed_model is None:
        print("\n" + "="*60)
        print("🔄 PLAN B: Création d'un nouveau modèle")
        new_model = create_new_model_architecture() 