import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.decomposition import PCA
import pickle

# Set your dataset path
dataset_path = r"Dataset"  # Change this to your actual Dataset path

def explore_deep_structure(dataset_path):
    """
    Deep explore the dataset structure to find where images are stored
    """
    dataset_path = Path(dataset_path)
    
    print("="*50)
    print("DEEP DATASET EXPLORATION")
    print("="*50)
    
    # Check skin types folder structure
    skin_types_path = dataset_path / "dataset_skintype_vit_final_crop"
    
    if skin_types_path.exists():
        print(f"\n📁 Exploring: {skin_types_path}")
        
        # Go through train, test, valid folders
        for subfolder in skin_types_path.iterdir():
            if subfolder.is_dir():
                print(f"\n  📁 {subfolder.name}/")
                
                # Check if there are subfolders inside train/test/valid
                inner_folders = [f for f in subfolder.iterdir() if f.is_dir()]
                
                if inner_folders:
                    print(f"     Found {len(inner_folders)} skin type categories:")
                    for inner in inner_folders:
                        # Count images in this category
                        img_count = len(list(inner.glob("*.[jJ][pP][gG]")) + 
                                      list(inner.glob("*.[pP][nN][gG]")) + 
                                      list(inner.glob("*.[jJ][pP][eE][gG]")))
                        print(f"       - {inner.name}: {img_count} images")
                else:
                    # If no inner folders, count images directly
                    img_count = len(list(subfolder.glob("*.[jJ][pP][gG]")) + 
                                  list(subfolder.glob("*.[pP][nN][gG]")))
                    print(f"     Images: {img_count}")
    
    # Check skin conditions folder structure
    skin_conditions_path = dataset_path / "Skin_Conditions"
    
    if skin_conditions_path.exists():
        print(f"\n📁 Exploring: {skin_conditions_path}")
        
        for subfolder in skin_conditions_path.iterdir():
            if subfolder.is_dir():
                img_count = len(list(subfolder.glob("*.[jJ][pP][gG]")) + 
                              list(subfolder.glob("*.[pP][nN][gG]")) + 
                              list(subfolder.glob("*.[jJ][pP][eE][gG]")))
                print(f"  - {subfolder.name}: {img_count} images")

def display_skin_types(dataset_path, img_size=(128, 128)):
    """
    Display skin types from train/test/valid folders
    """
    skin_types_path = Path(dataset_path) / "dataset_skintype_vit_final_crop"
    
    if not skin_types_path.exists():
        print(f"❌ Skin types folder not found at: {skin_types_path}")
        return
    
    # Collect all skin type categories from train, test, valid
    skin_type_categories = {}
    
    for split_folder in ['train', 'test', 'valid']:
        split_path = skin_types_path / split_folder
        if split_path.exists():
            # Check if there are category folders inside
            categories = [f for f in split_path.iterdir() if f.is_dir()]
            
            if categories:
                for category in categories:
                    if category.name not in skin_type_categories:
                        skin_type_categories[category.name] = []
                    
                    # Get images from this category
                    images = list(category.glob("*.[jJ][pP][gG]")) + \
                            list(category.glob("*.[pP][nN][gG]")) + \
                            list(category.glob("*.[jJ][pP][eE][gG]"))
                    
                    skin_type_categories[category.name].extend(images)
    
    print(f"\n📊 Found {len(skin_type_categories)} skin type categories:")
    for category, images in skin_type_categories.items():
        print(f"  - {category}: {len(images)} images")
    
    # Display 4 skin type categories (or all if less than 4)
    num_to_show = min(4, len(skin_type_categories))
    
    if num_to_show > 0:
        fig1, axes1 = plt.subplots(1, num_to_show, figsize=(16, 4))
        fig1.suptitle('Skin Types', fontsize=16, fontweight='bold')
        
        for i, (category, images) in enumerate(list(skin_type_categories.items())[:num_to_show]):
            if images:
                img = cv2.imread(str(images[0]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes1[i].imshow(img)
                    axes1[i].set_title(f"{category}\n({len(images)} images)", fontsize=12)
                else:
                    axes1[i].set_title(f"{category}\n(Error loading image)")
            else:
                axes1[i].set_title(f"{category}\n(No images found)")
            
            axes1[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("\n No skin type images found!")

def display_skin_conditions(dataset_path):
    """
    Display skin conditions from Skin_Conditions folder
    """
    skin_conditions_path = Path(dataset_path) / "Skin_Conditions"
    
    if not skin_conditions_path.exists():
        print(f" Skin conditions folder not found at: {skin_conditions_path}")
        return
    
    # Get all condition folders
    condition_folders = [f for f in skin_conditions_path.iterdir() if f.is_dir()]
    
    print(f"\n Found {len(condition_folders)} skin condition categories:")
    conditions_with_images = []
    
    for folder in condition_folders:
        images = list(folder.glob("*.[jJ][pP][gG]")) + \
                list(folder.glob("*.[pP][nN][gG]")) + \
                list(folder.glob("*.[jJ][pP][eE][gG]"))
        
        if images:
            conditions_with_images.append((folder.name, images))
            print(f"  - {folder.name}: {len(images)} images")
        else:
            print(f"  - {folder.name}: 0 images")
    
    # Display 6 skin conditions (or all if less than 6)
    num_to_show = min(6, len(conditions_with_images))
    
    if num_to_show > 0:
        rows = 2
        cols = 3
        fig2, axes2 = plt.subplots(rows, cols, figsize=(15, 10))
        fig2.suptitle('Skin Conditions', fontsize=16, fontweight='bold')
        
        for i in range(num_to_show):
            row = i // cols
            col = i % cols
            condition_name, images = conditions_with_images[i]
            
            if images:
                img = cv2.imread(str(images[0]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes2[row, col].imshow(img)
                    axes2[row, col].set_title(f"{condition_name}\n({len(images)} images)", fontsize=10)
                else:
                    axes2[row, col].set_title(f"{condition_name}\n(Error loading image)")
            else:
                axes2[row, col].set_title(f"{condition_name}\n(No images found)")
            
            axes2[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_to_show, rows * cols):
            row = i // cols
            col = i % cols
            axes2[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("\n No skin condition images found!")

def load_and_preprocess_images(dataset_path, img_size=(128, 128), grayscale=True):
    """
    Load images, resize, convert to grayscale, and prepare for training
    """
    skin_types_path = Path(dataset_path) / "dataset_skintype_vit_final_crop"
    
    if not skin_types_path.exists():
        print(f" Skin types folder not found at: {skin_types_path}")
        return None, None
    
    # Collect all skin type categories
    X = []  # Features (images)
    y = []  # Labels
    
    print("\n Loading and preprocessing images...")
    print(f"   Image size: {img_size}")
    print(f"   Grayscale: {grayscale}")
    
    for split_folder in ['train', 'test', 'valid']:
        split_path = skin_types_path / split_folder
        if split_path.exists():
            categories = [f for f in split_path.iterdir() if f.is_dir()]
            
            for category in categories:
                # Get all images in this category
                images = list(category.glob("*.[jJ][pP][gG]")) + \
                        list(category.glob("*.[pP][nN][gG]")) + \
                        list(category.glob("*.[jJ][pP][eE][gG]"))
                
                print(f"   Loading {len(images)} images from {category.name}...")
                
                for img_path in images:
                    try:
                        # Read image
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            # Resize image
                            img = cv2.resize(img, img_size)
                            
                            # Convert to grayscale if specified
                            if grayscale:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                # Add channel dimension back
                                img = img.reshape(img_size[0], img_size[1], 1)
                            
                            # Normalize pixel values (0-1)
                            img = img / 255.0
                            
                            # Flatten the image for Gradient Boosting
                            img_flattened = img.flatten()
                            
                            X.append(img_flattened)
                            y.append(category.name)
                    except Exception as e:
                        print(f"     Error loading {img_path.name}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n Total images loaded: {len(X)}")
    print(f"   Feature dimension: {X.shape[1]}")
    
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    Split data into train, validation, and test sets
    """
    print("\n" + "="*50)
    print("DATA SPLITTING")
    print("="*50)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Display class distribution
    print("\n Class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count} images")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_gradient_boosting(X_train, y_train, X_val, y_val, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Train Gradient Boosting classifier
    """
    print("\n" + "="*50)
    print("TRAINING GRADIENT BOOSTING CLASSIFIER")
    print("="*50)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Reduce dimensionality with PCA if needed (for high-dimensional image data)
    print(f"Original feature dimension: {X_train.shape[1]}")
    
    # Apply PCA for dimensionality reduction
    n_components = min(150, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    print(f"Reduced feature dimension: {X_train_pca.shape[1]}")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    # Create and train Gradient Boosting
    print(f"\nTraining Gradient Boosting with {n_estimators} estimators...")
    print(f"Learning rate: {learning_rate}")
    print(f"Max depth: {max_depth}")
    
    gb_classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        verbose=1
    )
    
    gb_classifier.fit(X_train_pca, y_train_encoded)
    
    # Evaluate on validation set
    y_val_pred = gb_classifier.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    
    print(f"\n Training completed!")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    
    return gb_classifier, label_encoder, pca

def evaluate_model(model, X_test, y_test, label_encoder, pca):
    """
    Evaluate the trained model on test data
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Transform test data using PCA
    X_test_pca = pca.transform(X_test)
    
    # Encode test labels
    y_test_encoded = label_encoder.transform(y_test)
    
    # Make predictions
    y_pred = model.predict(X_test_pca)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Gradient Boosting Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return accuracy, y_pred

def display_sample_predictions(model, X_test, y_test, label_encoder, pca, img_size=(128, 128), num_samples=6):
    """
    Display sample test images with their predictions
    """
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    # Transform test data using PCA
    X_test_pca = pca.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_pca)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Display sample predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sample Predictions - Gradient Boosting', fontsize=16, fontweight='bold')
    
    for i in range(min(num_samples, len(X_test))):
        row = i // 3
        col = i % 3
        
        # Reshape the flattened image back to original dimensions
        img_flat = X_test[i]
        # Calculate expected flattened size
        expected_size = img_size[0] * img_size[1]
        
        if len(img_flat) == expected_size:
            # Grayscale image
            img = img_flat.reshape(img_size[0], img_size[1])
            axes[row, col].imshow(img, cmap='gray')
        else:
            # Handle if image dimensions don't match
            img = img_flat[:expected_size].reshape(img_size[0], img_size[1])
            axes[row, col].imshow(img, cmap='gray')
        
        # Get actual and predicted labels
        actual = y_test[i]
        predicted = y_pred_labels[i]
        
        # Set title with colors
        if actual == predicted:
            title_color = 'green'
            title = f"Actual: {actual}\nPredicted: {predicted} ✓"
        else:
            title_color = 'red'
            title = f"Actual: {actual}\nPredicted: {predicted} ✗"
        
        axes[row, col].set_title(title, color=title_color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_feature_importance(model, pca, n_top_features=20):
    """
    Display feature importance from Gradient Boosting model
    """
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    
    # Get feature importance from Gradient Boosting
    feature_importance = model.feature_importances_
    
    # Plot top feature importances
    plt.figure(figsize=(12, 6))
    indices = np.argsort(feature_importance)[-n_top_features:]  # Top n features
    
    plt.barh(range(n_top_features), feature_importance[indices])
    plt.yticks(range(n_top_features), [f'PC{i+1}' for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {n_top_features} PCA Component Importances - Gradient Boosting')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 5 most important PCA components:")
    top_indices = np.argsort(feature_importance)[-5:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. PC{idx+1}: {feature_importance[idx]:.4f}")

def main():
    """
    Main function to execute the entire pipeline
    """
    # Update this path to your actual Dataset folder location
    dataset_path = r"Dataset"  # CHANGE THIS TO YOUR ACTUAL PATH
    
    # First explore the deep structure to understand where images are
    explore_deep_structure(dataset_path)
    
    # Display skin types
    print("\n" + "="*50)
    print("DISPLAYING SKIN TYPES")
    print("="*50)
    display_skin_types(dataset_path)
    
    # Display skin conditions
    print("\n" + "="*50)
    print("DISPLAYING SKIN CONDITIONS")
    print("="*50)
    display_skin_conditions(dataset_path)
    
    # Load and preprocess images (resize to 128x128 and convert to grayscale)
    print("\n" + "="*50)
    print("LOADING AND PREPROCESSING IMAGES")
    print("="*50)
    
    # Set image parameters
    IMG_SIZE = (128, 128)  # Resize to 128x128
    GRAYSCALE = True        # Convert to grayscale
    
    X, y = load_and_preprocess_images(dataset_path, img_size=IMG_SIZE, grayscale=GRAYSCALE)
    
    if X is None or len(X) == 0:
        print(" No images loaded. Please check your dataset path and structure.")
        return
    
    # Split data into train, validation, and test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, test_size=0.2, val_size=0.2)
    
    # Train Gradient Boosting classifier
    gb_model, label_encoder, pca = train_gradient_boosting(
        X_train, y_train, X_val, y_val, 
        n_estimators=100,      # Number of boosting stages
        learning_rate=0.1,     # Learning rate
        max_depth=3            # Maximum depth of individual trees
    )
    
    # Evaluate the model
    test_accuracy, y_pred = evaluate_model(gb_model, X_test, y_test, label_encoder, pca)
    
    # Display sample predictions
    display_sample_predictions(gb_model, X_test, y_test, label_encoder, pca, img_size=IMG_SIZE, num_samples=6)
    
    # Display feature importance
    display_feature_importance(gb_model, pca, n_top_features=20)
  
    # Additional performance metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Algorithm: Gradient Boosting Classifier")
    print(f"Number of estimators: 100")
    print(f"Learning rate: 0.1")
    print(f"Max depth: 3")
    print(f"PCA components: {pca.n_components_}")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    print("\n Pipeline completed successfully!")
    
    # Save the model (NOW INSIDE THE MAIN FUNCTION)
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    try:
        with open('gradient_boosting_model.pkl', 'wb') as f:
            pickle.dump((gb_model, label_encoder, pca), f)
        print(" Model saved successfully as 'gradient_boosting_model.pkl'")
        print(f"   Model type: {type(gb_model)}")
        print(f"   Label encoder classes: {label_encoder.classes_}")
        print(f"   PCA components: {pca.n_components_}")
    except Exception as e:
        print(f" Error saving model: {e}")
    
    # Also save the model with a timestamp for backup
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f'gradient_boosting_model_{timestamp}.pkl'
    try:
        with open(backup_filename, 'wb') as f:
            pickle.dump((gb_model, label_encoder, pca), f)
        print(f" Backup model saved as '{backup_filename}'")
    except Exception as e:
        print(f" Could not save backup: {e}")

if __name__ == "__main__":
    main()