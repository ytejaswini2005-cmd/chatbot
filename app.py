from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import pickle
import cv2
import numpy as np
from pathlib import Path
from functools import wraps
import traceback
import sys

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = 'gradient_boosting_model.pkl'
IMG_SIZE = (128, 128)

# Global variables for model
gb_model = None
label_encoder = None
pca = None
model_loaded = False

def load_model():
    """Load the trained model with comprehensive error handling"""
    global gb_model, label_encoder, pca, model_loaded
    
    print("\n" + "="*50)
    print("LOADING MODEL...")
    print("="*50)
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found at: {MODEL_PATH}")
            model_loaded = False
            return False
        
        print(f"📁 Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, tuple) and len(model_data) == 3:
            gb_model, label_encoder, pca = model_data
            print(f"✅ Model loaded successfully!")
            print(f"   Label encoder classes: {list(label_encoder.classes_)}")
            model_loaded = True
            return True
        else:
            print(f"❌ Unexpected model format")
            model_loaded = False
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        model_loaded = False
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    global pca, IMG_SIZE, model_loaded
    
    if not model_loaded or pca is None:
        print("   ❌ Model not loaded. Cannot preprocess image.")
        return None
    
    try:
        print(f"\n🔍 Processing image: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"   ❌ File does not exist: {image_path}")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ OpenCV failed to read image.")
            return None
        
        print(f"   Original image shape: {img.shape}")
        
        # Resize image
        img = cv2.resize(img, IMG_SIZE)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values (0-1)
        img = img.astype(np.float32) / 255.0
        
        # Flatten the image
        img_flattened = img.flatten()
        img_reshaped = img_flattened.reshape(1, -1)
        
        # Apply PCA
        img_pca = pca.transform(img_reshaped)
        print(f"   PCA transformed shape: {img_pca.shape}")
        
        return img_pca
        
    except Exception as e:
        print(f"   ❌ Error in preprocess_image: {e}")
        traceback.print_exc()
        return None
def get_skin_type_recommendations(skin_type):
    """Get detailed skincare recommendations based on skin type"""
    recommendations_db = {
        'oily': {
            'description': 'Oily skin produces excess sebum, leading to shine, enlarged pores, and potential breakouts.',
            'suggestions': [
                "Use oil-free, non-comedogenic products to prevent clogged pores",
                "Cleanse twice daily with a gentle foaming cleanser containing salicylic acid",
                "Use a lightweight, oil-free moisturizer even if your skin feels oily",
                "Apply a clay mask 2-3 times a week to absorb excess oil",
                "Use blotting papers during the day to control shine without disturbing makeup",
                "Avoid harsh scrubs that can over-stimulate oil production",
                "Use a gel-based sunscreen with at least SPF 30",
                "Incorporate niacinamide to regulate sebum production",
                "Use chemical exfoliants like salicylic acid 2-3 times a week",
                "Avoid touching your face throughout the day",
                "Keep your hair away from your face as hair products can clog pores",
                "Use a gentle toner with witch hazel to balance pH",
                "Consider using retinoids at night to prevent breakouts",
                "Change pillowcases weekly to prevent bacterial buildup",
                "Stay hydrated and maintain a balanced diet low in processed foods"
            ],
            'products': [
                "CeraVe Foaming Facial Cleanser",
                "La Roche-Posay Effaclar Mat Oil-Free Mattifying Moisturizer",
                "The Ordinary Niacinamide 10% + Zinc 1%",
                "Paula's Choice Skin Perfecting 2% BHA Liquid Exfoliant",
                "Innisfree Jeju Volcanic Pore Clay Mask",
                "Neutrogena Hydro Boost Water Gel",
                "COSRX Salicylic Acid Daily Gentle Cleanser",
                "Supergoop! Unseen Sunscreen SPF 40",
                "Kiehl's Rare Earth Deep Pore Cleansing Masque",
                "Drunk Elephant T.L.C. Framboos Glycolic Night Serum",
                "Origins Zero Oil Pore Purifying Toner",
                "Glossier Solution Exfoliating Skin Perfector",
                "Mario Badescu Special Healing Powder",
                "The Face Shop Rice Water Bright Foaming Cleanser",
                "Biore Deep Pore Charcoal Cleanser"
            ],
            'routine': [
                "Morning: Gentle foaming cleanser → Oil-free moisturizer → Mattifying sunscreen SPF 30+",
                "Evening: Double cleanse → BHA exfoliant (2-3x/week) → Lightweight moisturizer",
                "Weekly: Apply clay mask 2-3 times a week",
                "Spot Treatment: Apply salicylic acid spot treatment on breakouts",
                "Morning Step 1: Wash face with lukewarm water and gentle foaming cleanser",
                "Morning Step 2: Apply toner with salicylic acid",
                "Morning Step 3: Apply oil-free serum with niacinamide",
                "Morning Step 4: Use lightweight gel moisturizer",
                "Morning Step 5: Apply oil-free sunscreen SPF 30+",
                "Evening Step 1: Remove makeup with micellar water",
                "Evening Step 2: Cleanse with foaming cleanser",
                "Evening Step 3: Apply BHA exfoliant (on exfoliation nights)",
                "Evening Step 4: Apply treatment serum (retinol or niacinamide)",
                "Evening Step 5: Moisturize with lightweight night cream"
            ]
        },
        'dry': {
            'description': 'Dry skin lacks moisture, often feeling tight, flaky, and may show fine lines.',
            'suggestions': [
                "Use cream or oil-based cleansers that don't strip natural oils",
                "Apply moisturizer immediately after cleansing while skin is still damp",
                "Use hydrating serums with hyaluronic acid and glycerin",
                "Apply facial oils like jojoba or rosehip oil for extra hydration",
                "Use a humidifier in your bedroom to add moisture to the air",
                "Avoid hot water when washing your face as it strips natural oils",
                "Use gentle, fragrance-free products to prevent irritation",
                "Apply overnight masks 2-3 times a week for intense hydration",
                "Exfoliate gently with lactic acid instead of harsh scrubs",
                "Layer products from thinnest to thickest consistency",
                "Use a creamy, hydrating sunscreen daily",
                "Avoid alcohol-based toners that can dry out skin",
                "Apply a rich eye cream to prevent fine lines",
                "Drink plenty of water throughout the day",
                "Use a silk pillowcase to prevent moisture loss"
            ],
            'products': [
                "CeraVe Hydrating Facial Cleanser",
                "La Roche-Posay Lipikar AP+ Moisturizing Cream",
                "The Ordinary Hyaluronic Acid 2% + B5",
                "Kiehl's Ultra Facial Cream",
                "CeraVe Moisturizing Cream",
                "Laneige Water Sleeping Mask",
                "Rosehip Oil - Natural facial oil",
                "Drunk Elephant Lala Retro Whipped Cream",
                "First Aid Beauty Ultra Repair Cream",
                "Kiehl's Midnight Recovery Concentrate",
                "Aveeno Eczema Therapy Moisturizing Cream",
                "Vanicream Moisturizing Cream",
                "The Inkey List Squalane Oil",
                "Weleda Skin Food",
                "COSRX Advanced Snail 92 All in One Cream"
            ],
            'routine': [
                "Morning: Cream cleanser → Hydrating serum → Rich moisturizer → Moisturizing sunscreen",
                "Evening: Oil-based cleanser → Cream cleanser → Hydrating toner → Facial oil → Night cream",
                "Weekly: Apply hydrating sheet masks 2-3 times a week",
                "Exfoliate gently with lactic acid once a week",
                "Use overnight sleeping masks for intense hydration",
                "Morning Step 1: Cleanse with cream or milk cleanser",
                "Morning Step 2: Apply hydrating toner with rose water",
                "Morning Step 3: Use hyaluronic acid serum",
                "Morning Step 4: Apply rich moisturizer with ceramides",
                "Morning Step 5: Use moisturizing sunscreen SPF 30+",
                "Evening Step 1: Remove makeup with cleansing oil",
                "Evening Step 2: Cleanse with cream cleanser",
                "Evening Step 3: Apply hydrating essence",
                "Evening Step 4: Apply facial oil (rosehip or squalane)",
                "Evening Step 5: Apply rich night cream"
            ]
        },
        'normal': {
            'description': 'Normal skin is well-balanced, not too oily or dry, with minimal imperfections.',
            'suggestions': [
                "Maintain a consistent skincare routine with gentle products",
                "Use a gentle, pH-balanced cleanser twice daily",
                "Apply lightweight moisturizer to maintain skin barrier",
                "Use sunscreen daily to prevent premature aging",
                "Incorporate antioxidants like vitamin C in your morning routine",
                "Exfoliate 1-2 times a week to maintain smooth texture",
                "Use a hydrating serum to boost moisture",
                "Apply eye cream to prevent fine lines",
                "Get adequate sleep to allow skin to regenerate",
                "Stay hydrated by drinking 8+ glasses of water daily",
                "Use a silk pillowcase to prevent friction",
                "Maintain a balanced diet rich in fruits and vegetables",
                "Avoid touching your face to prevent bacterial transfer",
                "Change pillowcases weekly",
                "Use a gentle toner to maintain pH balance"
            ],
            'products': [
                "CeraVe Hydrating Facial Cleanser",
                "Kiehl's Ultra Facial Cream",
                "The Ordinary Vitamin C Suspension 23%",
                "La Roche-Posay Anthelios Sunscreen SPF 50",
                "COSRX Advanced Snail 96 Mucin Power Essence",
                "Neutrogena Hydro Boost Water Gel",
                "Paula's Choice RESIST Anti-Aging Clear Skin Hydrator",
                "Glossier Milky Jelly Cleanser",
                "Drunk Elephant C-Firma Day Serum",
                "First Aid Beauty Face Cleanser",
                "Summer Fridays Cloud Dew Gel Cream",
                "Tatcha The Water Cream",
                "Biossance Squalane + Vitamin C Rose Oil",
                "Fresh Soy Face Cleanser",
                "Youth to the People Superfood Cleanser"
            ],
            'routine': [
                "Morning: Gentle cleanser → Vitamin C serum → Lightweight moisturizer → Sunscreen SPF 30+",
                "Evening: Gentle cleanser → Hydrating serum → Night cream",
                "Weekly: Exfoliate 1-2 times with gentle chemical exfoliant",
                "Apply sheet mask once a week for extra hydration",
                "Use eye cream morning and night",
                "Morning Step 1: Cleanse with gentle foaming cleanser",
                "Morning Step 2: Apply vitamin C serum",
                "Morning Step 3: Use lightweight moisturizer",
                "Morning Step 4: Apply broad spectrum sunscreen",
                "Evening Step 1: Double cleanse to remove sunscreen",
                "Evening Step 2: Apply hydrating toner",
                "Evening Step 3: Use peptide serum",
                "Evening Step 4: Apply night cream with retinol (2-3x/week)",
                "Evening Step 5: Apply eye cream"
            ]
        },
        'combination': {
            'description': 'Combination skin has both oily (typically T-zone) and dry areas.',
            'suggestions': [
                "Use different products for different areas of your face",
                "Use a gentle, balanced cleanser that doesn't strip or over-moisturize",
                "Apply lightweight moisturizer everywhere, but use richer products on dry areas",
                "Use oil-control products on T-zone only",
                "Hydrate dry areas with richer creams",
                "Use clay masks on T-zone, hydrating masks on cheeks",
                "Avoid heavy creams on oily areas",
                "Use gel-based products for oily zones",
                "Use cream-based products for dry zones",
                "Blot T-zone during the day without disturbing makeup",
                "Use a balancing toner to regulate oil production",
                "Apply hyaluronic acid for overall hydration",
                "Use a gentle exfoliant 2-3 times a week",
                "Keep separate products for different zones",
                "Use lightweight, non-comedogenic sunscreen"
            ],
            'products': [
                "COSRX Low pH Good Morning Gel Cleanser",
                "Kiehl's Ultra Facial Oil-Free Gel Cream",
                "The Ordinary Niacinamide 10% + Zinc 1%",
                "La Roche-Posay Toleriane Double Repair Face Moisturizer",
                "Paula's Choice Skin Balancing Pore-Reducing Toner",
                "Dr. Jart+ Teatreement Moisturizer",
                "Glow Recipe Watermelon Glow PHA+BHA Pore-Tight Toner",
                "Cetaphil Pro Oil Absorbing Moisturizer",
                "Neutrogena Hydro Boost Gel-Cream",
                "Fresh Rose Deep Hydration Face Cream",
                "Origins Zero Oil Pore Purifying Toner",
                "Belif The True Cream Aqua Bomb",
                "Laneige Cream Skin Refiner",
                "Tatcha The Water Cream",
                "Peter Thomas Roth Water Drench Cloud Cream"
            ],
            'routine': [
                "Morning: Gentle cleanser → Balancing toner → Lightweight moisturizer (full face) → Sunscreen",
                "Evening: Double cleanse → BHA on T-zone → Hydrating serum on cheeks → Night cream",
                "Weekly: Clay mask on T-zone, hydrating mask on cheeks",
                "Use oil-free products on forehead, nose, and chin",
                "Use richer creams on cheeks and around eyes",
                "Morning Step 1: Cleanse with gel cleanser",
                "Morning Step 2: Apply balancing toner",
                "Morning Step 3: Use lightweight moisturizer on T-zone",
                "Morning Step 4: Apply richer cream on cheeks",
                "Morning Step 5: Use oil-free sunscreen",
                "Evening Step 1: Remove makeup with micellar water",
                "Evening Step 2: Cleanse with gel cleanser",
                "Evening Step 3: Apply BHA on T-zone only",
                "Evening Step 4: Use hydrating serum on cheeks",
                "Evening Step 5: Apply lightweight night cream"
            ]
        }
    }
    
    skin_type_lower = skin_type.lower()
    
    # Check for exact match
    if skin_type_lower in recommendations_db:
        return recommendations_db[skin_type_lower]
    
    # Handle alternative names
    if 'oily' in skin_type_lower:
        return recommendations_db['oily']
    elif 'dry' in skin_type_lower:
        return recommendations_db['dry']
    elif 'normal' in skin_type_lower:
        return recommendations_db['normal']
    elif 'combo' in skin_type_lower or 'combination' in skin_type_lower:
        return recommendations_db['combination']
    else:
        return recommendations_db['normal']


# Add this function to get skin condition recommendations
def get_skin_condition_recommendations(condition):
    """Get detailed recommendations based on skin condition"""
    conditions_db = {
        'acne': {
            'description': 'Acne is a common skin condition characterized by pimples, blackheads, and inflamed lesions caused by clogged pores and bacterial growth.',
            'suggestions': [
                "Use non-comedogenic products that won't clog pores",
                "Cleanse twice daily with gentle, acne-fighting cleanser",
                "Never pop or squeeze pimples - this can cause scarring",
                "Change pillowcases every 2-3 days to prevent bacterial buildup",
                "Keep hair away from face as hair products can trigger breakouts",
                "Use oil-free, mattifying sunscreen daily",
                "Avoid touching your face throughout the day",
                "Use salicylic acid or benzoyl peroxide spot treatments",
                "Incorporate niacinamide to reduce inflammation",
                "Consider seeing a dermatologist for persistent acne",
                "Use gentle, alcohol-free products to prevent irritation",
                "Avoid harsh physical scrubs that can aggravate acne",
                "Keep stress levels in check as stress can trigger breakouts",
                "Stay hydrated and maintain a balanced diet",
                "Don't over-wash - twice daily is sufficient"
            ],
            'products': [
                "CeraVe Acne Foaming Cream Cleanser",
                "La Roche-Posay Effaclar Duo Acne Treatment",
                "The Ordinary Salicylic Acid 2% Solution",
                "Paula's Choice Clear Acne Kit",
                "Neutrogena Rapid Clear Stubborn Acne Spot Gel",
                "Differin Adapalene Gel 0.1%",
                "COSRX Acne Pimple Master Patch",
                "Mario Badescu Drying Lotion",
                "Cetaphil Gentle Skin Cleanser",
                "Aztec Secret Indian Healing Clay Mask",
                "The Inkey List Succinic Acid Acne Treatment",
                "Drunk Elephant T.L.C. Framboos Glycolic Night Serum",
                "Kiehl's Blue Herbal Spot Treatment",
                "Sunday Riley UFO Ultra-Clarifying Face Oil",
                "Benzac AC 5% Benzoyl Peroxide Gel"
            ],
            'routine': [
                "Morning: Gentle cleanser → Acne treatment → Oil-free moisturizer → Mattifying sunscreen",
                "Evening: Double cleanse → Chemical exfoliant → Spot treatment → Lightweight moisturizer",
                "Weekly: Apply clay mask 2 times a week to draw out impurities",
                "Use hydrocolloid patches on active breakouts overnight",
                "Morning Step 1: Cleanse with gentle, non-stripping cleanser",
                "Morning Step 2: Apply benzoyl peroxide or salicylic acid treatment",
                "Morning Step 3: Use lightweight, oil-free moisturizer",
                "Morning Step 4: Apply non-comedogenic sunscreen SPF 30+",
                "Evening Step 1: Remove makeup with micellar water",
                "Evening Step 2: Cleanse with acne-fighting cleanser",
                "Evening Step 3: Apply retinol or adapalene (start slowly)",
                "Evening Step 4: Apply spot treatment on active pimples",
                "Evening Step 5: Moisturize with gentle, hydrating moisturizer"
            ],
            'doctors_routine': [
                "Consult a dermatologist for prescription-strength treatments",
                "Consider oral antibiotics for moderate to severe acne",
                "Professional chemical peels to unclog pores",
                "Extraction procedures by professionals for blackheads",
                "Light therapy treatments to kill acne-causing bacteria",
                "Microneedling to improve acne scars",
                "Corticosteroid injections for large, painful cysts",
                "Prescription topical retinoids like tretinoin",
                "Birth control pills for hormonal acne in women",
                "Spironolactone for hormonal acne treatment",
                "Isotretinoin (Accutane) for severe, resistant acne",
                "Regular follow-up appointments every 4-6 weeks",
                "Patch testing before starting new treatments",
                "Complete full course of prescribed medications",
                "Document progress with monthly photos"
            ],
            'food_intake': [
                "🥬 Leafy Greens: Spinach, kale, Swiss chard - rich in antioxidants",
                "🫐 Berries: Blueberries, strawberries, raspberries - anti-inflammatory",
                "🍵 Green Tea: Contains antioxidants that reduce inflammation",
                "🐟 Fatty Fish: Salmon, mackerel, sardines - omega-3 fatty acids",
                "🌰 Nuts and Seeds: Almonds, walnuts, chia seeds - zinc and vitamin E",
                "🟡 Turmeric: Anti-inflammatory properties",
                " Ginger: Reduces inflammation and boosts immunity",
                "🌶️ Colorful Vegetables: Bell peppers, carrots, sweet potatoes",
                "🍎 Low-Glycemic Fruits: Apples, pears, citrus fruits",
                "🥛 Probiotic Foods: Yogurt, kefir, kimchi - gut health",
                "🌾 Whole Grains: Oats, quinoa, brown rice",
                " Legumes: Lentils, chickpeas, beans",
                "🥑 Avocados: Healthy fats for skin health",
                "🍅 Tomatoes: Rich in lycopene",
                "💧 Water: Stay hydrated with at least 8 glasses daily"
            ]
        },
        'eczema': {
            'description': 'Eczema (atopic dermatitis) is a chronic condition causing dry, itchy, and inflamed skin patches.',
            'suggestions': [
                "Use fragrance-free, gentle products to avoid irritation",
                "Apply moisturizer immediately after bathing to lock in moisture",
                "Take short, lukewarm showers instead of hot baths",
                "Use a humidifier to add moisture to the air",
                "Wear soft, breathable fabrics like cotton",
                "Avoid scratching - use cold compresses for itching",
                "Identify and avoid triggers like stress and allergens",
                "Use gentle, non-soap cleansers",
                "Apply prescription medications as directed",
                "Keep nails short to prevent skin damage",
                "Use wet wrap therapy for severe flare-ups",
                "Avoid wool and synthetic fabrics",
                "Use mild, fragrance-free laundry detergent",
                "Avoid sudden temperature changes",
                "Manage stress through meditation or exercise"
            ],
            'products': [
                "CeraVe Healing Ointment",
                "La Roche-Posay Lipikar Balm AP+",
                "Aveeno Eczema Therapy Moisturizing Cream",
                "Vanicream Moisturizing Cream",
                "Cetaphil Restoraderm Eczema Calming Body Wash",
                "Eucerin Eczema Relief Cream",
                "Aquaphor Healing Ointment",
                "Gladskin Eczema Cream",
                "Bioderma Atoderm Intensive Baume",
                "Mustela Stelatopia Emollient Cream",
                "CeraVe Itch Relief Moisturizing Cream",
                "First Aid Beauty Ultra Repair Cream",
                "Skinfix Eczema+ Dermatitis Face Balm",
                "Tower 28 Beauty SOS Daily Rescue Facial Spray",
                "Avene XeraCalm A.D Lipid-Replenishing Cream"
            ],
            'routine': [
                "Morning: Gentle cleanser → Thick moisturizer → Sunscreen",
                "Evening: Gentle cleanser → Prescription cream → Rich moisturizer",
                "Weekly: Apply emollient-rich masks for intense hydration",
                "Use wet wrap therapy for severe flare-ups",
                "Morning Step 1: Cleanse with gentle, non-soap cleanser",
                "Morning Step 2: Apply thick moisturizer while skin is damp",
                "Morning Step 3: Use mineral-based sunscreen",
                "Evening Step 1: Gentle cleanse with lukewarm water",
                "Evening Step 2: Apply prescription cream (if prescribed)",
                "Evening Step 3: Apply rich ointment or cream",
                "Evening Step 4: Use humidifier in bedroom"
            ],
            'doctors_routine': [
                "Consult a dermatologist for proper diagnosis and treatment plan",
                "Topical corticosteroids for flare-ups",
                "Topical calcineurin inhibitors for sensitive areas",
                "Oral antihistamines to reduce itching",
                "Phototherapy (light therapy) for moderate to severe cases",
                "Systemic immunosuppressants for severe eczema",
                "Biologic medications like dupilumab",
                "Allergy testing to identify triggers",
                "Patch testing for contact allergies",
                "Regular moisturizing with medical-grade emollients",
                "Wet wrap therapy under medical supervision",
                "Bleach baths to reduce bacterial infection risk",
                "Crisaborole ointment for mild to moderate eczema",
                "JAK inhibitors for treatment-resistant cases",
                "Regular follow-up every 3-6 months"
            ],
            'food_intake': [
                "🐟 Omega-3 Rich Fish: Salmon, tuna, sardines - anti-inflammatory",
                "🥑 Avocados: Rich in healthy fats and vitamin E",
                "🥕 Carrots: Beta-carotene for skin health",
                "🥦 Broccoli: Vitamin C and antioxidants",
                "🍠 Sweet Potatoes: Vitamin A for skin repair",
                "🥬 Spinach: Iron and vitamins for skin health",
                "🫐 Blueberries: Antioxidants reduce inflammation",
                "🍎 Apples: Quercetin reduces allergic reactions",
                "🍌 Bananas: Rich in potassium and vitamin B6",
                "🌰 Walnuts: Omega-3 fatty acids",
                "🥛 Probiotic Yogurt: Supports gut health",
                "🍵 Green Tea: Anti-inflammatory properties",
                "🌾 Oats: Anti-inflammatory and soothing",
                "💧 Water: Essential for hydration",
                "🥥 Coconut Oil: Natural moisturizer when consumed"
            ]
        },
        'psoriasis': {
            'description': 'Psoriasis is an autoimmune condition causing rapid skin cell growth, resulting in thick, scaly patches.',
            'suggestions': [
                "Use thick, fragrance-free moisturizers daily",
                "Take lukewarm baths with Epsom salts or oatmeal",
                "Use gentle, non-irritating cleansers",
                "Avoid scratching - use cold compresses for itching",
                "Identify and avoid triggers like stress and infections",
                "Use prescribed medications consistently",
                "Protect skin from cuts and injuries",
                "Avoid alcohol as it can trigger flare-ups",
                "Maintain a healthy weight",
                "Get moderate sunlight exposure (with protection)",
                "Use humidifiers to prevent dry skin",
                "Avoid harsh soaps and hot water",
                "Wear soft, loose-fitting clothing",
                "Manage stress through relaxation techniques",
                "Quit smoking as it worsens psoriasis"
            ],
            'products': [
                "CeraVe Psoriasis Cream",
                "MG217 Medicated Psoriasis Cream",
                "Neutrogena T/Gel Therapeutic Shampoo",
                "Dermarest Psoriasis Medicated Shampoo",
                "Curél Hydra Therapy Wet Skin Moisturizer",
                "Eucerin Skin Calming Cream",
                "Aveeno Active Naturals Skin Relief Body Wash",
                "Gold Bond Multi-Symptom Psoriasis Relief Cream",
                "Vanicream Moisturizing Cream",
                "Bioderma Atoderm Intensive Baume",
                "La Roche-Posay Lipikar AP+ Balm",
                "Cetaphil Restoraderm Eczema Calming Lotion",
                "First Aid Beauty Ultra Repair Cream",
                "Aveeno Eczema Therapy Moisturizing Cream",
                "CeraVe Healing Ointment"
            ],
            'routine': [
                "Morning: Gentle cleanse → Thick moisturizer → Sunscreen",
                "Evening: Gentle cleanse → Prescription treatment → Rich moisturizer",
                "Weekly: Soak in oatmeal bath to soothe scaling",
                "Apply medicated creams immediately after bathing",
                "Morning Step 1: Cleanse with gentle, fragrance-free cleanser",
                "Morning Step 2: Apply thick moisturizer while skin is damp",
                "Morning Step 3: Use mineral sunscreen on exposed areas",
                "Evening Step 1: Lukewarm bath with Epsom salts",
                "Evening Step 2: Apply prescription topical medication",
                "Evening Step 3: Apply rich moisturizing cream",
                "Evening Step 4: Use humidifier in bedroom"
            ],
            'doctors_routine': [
                "Consult a dermatologist specializing in autoimmune conditions",
                "Topical corticosteroids for mild to moderate psoriasis",
                "Vitamin D analogues (calcipotriene) to slow cell growth",
                "Topical retinoids to reduce inflammation",
                "Phototherapy (UVB light) treatments 2-3 times weekly",
                "Systemic medications like methotrexate or cyclosporine",
                "Biologic drugs (Humira, Enbrel, Stelara) for moderate-severe cases",
                "Oral retinoids like acitretin",
                "Excimer laser for localized plaques",
                "Regular monitoring for psoriatic arthritis",
                "Lifestyle modifications for trigger management",
                "Stress management programs",
                "Smoking cessation programs",
                "Weight management support",
                "Regular follow-up every 3-6 months"
            ],
            'food_intake': [
                "🐟 Fatty Fish: Salmon, mackerel, sardines - omega-3 anti-inflammatory",
                "🥑 Avocados: Healthy fats and vitamin E",
                " Olive Oil: Anti-inflammatory properties",
                "🥬 Leafy Greens: Spinach, kale - rich in antioxidants",
                "🫐 Berries: Blueberries, strawberries - reduce inflammation",
                "🥕 Carrots: Beta-carotene for skin health",
                "🍠 Sweet Potatoes: Vitamin A and antioxidants",
                "🥦 Broccoli: Vitamin C and sulforaphane",
                "🌰 Walnuts: Omega-3 fatty acids",
                "🍵 Green Tea: Anti-inflammatory catechins",
                "🌾 Whole Grains: Oats, quinoa - reduce inflammation",
                "🧄 Garlic: Anti-inflammatory properties",
                "🟡 Turmeric: Powerful anti-inflammatory",
                "💧 Water: Essential for hydration",
                "🍋 Citrus Fruits: Vitamin C for immune support"
            ]
        }
    }
    
    condition_lower = condition.lower()
    
    # Check for exact match
    if condition_lower in conditions_db:
        return conditions_db[condition_lower]
    
    # Handle common variations
    for key in conditions_db.keys():
        if key in condition_lower:
            return conditions_db[key]
    
    # Default return for unknown conditions
    return {
        'description': 'Please consult a dermatologist for proper diagnosis and treatment.',
        'suggestions': [
            "Consult a dermatologist for accurate diagnosis",
            "Use gentle, fragrance-free skincare products",
            "Protect your skin from excessive sun exposure",
            "Stay hydrated by drinking plenty of water",
            "Maintain a healthy, balanced diet",
            "Get adequate sleep for skin repair",
            "Manage stress through relaxation techniques",
            "Avoid harsh chemicals and irritants",
            "Use lukewarm water instead of hot water",
            "Moisturize regularly to maintain skin barrier",
            "Avoid scratching affected areas",
            "Wear soft, breathable fabrics",
            "Keep skin clean but don't over-wash",
            "Take photos to track progress",
            "Follow prescribed treatments consistently"
        ],
        'products': [
            "CeraVe Hydrating Cleanser",
            "Cetaphil Gentle Skin Cleanser",
            "Vanicream Moisturizing Cream",
            "Aquaphor Healing Ointment",
            "La Roche-Posay Toleriane Double Repair Moisturizer",
            "Aveeno Calm + Restore Oat Gel Moisturizer",
            "First Aid Beauty Ultra Repair Cream",
            "Kiehl's Ultra Facial Cream",
            "The Ordinary Natural Moisturizing Factors",
            "Neutrogena Hydro Boost Water Gel"
        ],
        'routine': [
            "Morning: Gentle cleanse → Light moisturizer → Sunscreen",
            "Evening: Gentle cleanse → Treatment cream → Rich moisturizer",
            "Weekly: Gentle exfoliation if tolerated",
            "Morning Step 1: Cleanse with lukewarm water",
            "Morning Step 2: Apply moisturizer",
            "Morning Step 3: Use sunscreen SPF 30+",
            "Evening Step 1: Gentle cleanse",
            "Evening Step 2: Apply prescribed treatment",
            "Evening Step 3: Apply night cream"
        ],
        'doctors_routine': [
            "Schedule appointment with board-certified dermatologist",
            "Get proper diagnosis through physical examination",
            "Discuss treatment options and potential side effects",
            "Follow prescribed treatment plan consistently",
            "Attend follow-up appointments as scheduled",
            "Report any adverse reactions immediately",
            "Consider patch testing for allergies",
            "Document your symptoms and triggers",
            "Ask about lifestyle modifications",
            "Get second opinion if needed"
        ],
        'food_intake': [
            "🥬 Leafy Greens: Spinach, kale, Swiss chard",
            "🫐 Berries: Blueberries, strawberries, raspberries",
            "🐟 Fatty Fish: Salmon, mackerel, sardines",
            "🌰 Nuts and Seeds: Almonds, walnuts, chia seeds",
            "🥑 Avocados: Healthy fats",
            "🍵 Green Tea: Antioxidants",
            "🌾 Whole Grains: Oats, quinoa, brown rice",
            "🥕 Colorful Vegetables: Carrots, bell peppers",
            "🍎 Fresh Fruits: Apples, pears, citrus",
            "💧 Water: Stay hydrated"
        ]
    }

# Database functions
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            predicted_skin_type TEXT,
            confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized!")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session['username'])

@app.route('/data-explore', methods=['GET', 'POST'])
@login_required
def data_explore():
    global model_loaded
    
    prediction_result = None
    uploaded_image = None
    current_skin_type = None
    
    if not model_loaded:
        flash('⚠️ Model not loaded. Please check if model file exists.', 'danger')
        return render_template('data_explore.html', prediction=None, uploaded_image=None)
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print(f"📁 File saved: {filepath}")
                
                img_pca = preprocess_image(filepath)
                
                if img_pca is not None:
                    prediction_encoded = gb_model.predict(img_pca)
                    prediction_proba = gb_model.predict_proba(img_pca)
                    confidence = np.max(prediction_proba) * 100
                    predicted_skin_type = label_encoder.inverse_transform(prediction_encoded)[0]
                    print(f"🎯 Prediction: {predicted_skin_type} with confidence {confidence:.2f}%")
                    
                    # Get recommendations
                    recommendations = get_skin_type_recommendations(predicted_skin_type)
                    
                    # Store in session for button navigation
                    session['current_skin_type'] = predicted_skin_type
                    session['current_recommendations'] = recommendations
                    
                    # Store prediction in database
                    conn = get_db_connection()
                    conn.execute(
                        'INSERT INTO predictions (user_id, image_path, predicted_skin_type, confidence_score) VALUES (?, ?, ?, ?)',
                        (session['user_id'], filename, predicted_skin_type, confidence)
                    )
                    conn.commit()
                    conn.close()
                    
                    prediction_result = {
                        'skin_type': predicted_skin_type,
                        'confidence': f"{confidence:.2f}%",
                        'recommendations': recommendations
                    }
                    uploaded_image = filename
                    current_skin_type = predicted_skin_type
                    
                    flash(f'✅ Prediction complete! Your skin type is: {predicted_skin_type}', 'success')
                else:
                    flash('❌ Error processing image. Please ensure the image is clear and try again.', 'danger')
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                print(f"❌ Unexpected error: {e}")
        else:
            flash('Invalid file type! Please upload PNG, JPG, JPEG, or BMP files.', 'danger')
    
    return render_template('data_explore.html', 
                         prediction=prediction_result, 
                         uploaded_image=uploaded_image,
                         current_skin_type=current_skin_type)

@app.route('/suggestions')
@login_required
def suggestions():
    """Display suggestions page"""
    skin_type = session.get('current_skin_type')
    recommendations = session.get('current_recommendations')
    
    if not skin_type or not recommendations:
        flash('Please upload an image first to get suggestions.', 'warning')
        return redirect(url_for('data_explore'))
    
    return render_template('suggestions.html', 
                         skin_type=skin_type, 
                         suggestions=recommendations['suggestions'],
                         description=recommendations['description'])

@app.route('/products')
@login_required
def products():
    """Display products page"""
    skin_type = session.get('current_skin_type')
    recommendations = session.get('current_recommendations')
    
    if not skin_type or not recommendations:
        flash('Please upload an image first to see product recommendations.', 'warning')
        return redirect(url_for('data_explore'))
    
    # Debug print to verify data
    print(f"Skin Type: {skin_type}")
    print(f"Products: {recommendations['products']}")
    print(f"Number of products: {len(recommendations['products'])}")
    
    return render_template('products.html', 
                         skin_type=skin_type, 
                         products=recommendations['products'])

@app.route('/routine')
@login_required
def routine():
    """Display daily routine page"""
    skin_type = session.get('current_skin_type')
    recommendations = session.get('current_recommendations')
    
    if not skin_type or not recommendations:
        flash('Please upload an image first to see skincare routine.', 'warning')
        return redirect(url_for('data_explore'))
    
    return render_template('routine.html', 
                         skin_type=skin_type, 
                         routine=recommendations['routine'],
                         description=recommendations['description'])

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    global model_loaded
    
    prediction_result = None
    uploaded_image = None
    
    if not model_loaded:
        flash('⚠️ Model not loaded. Please contact administrator.', 'danger')
        return render_template('prediction.html', prediction=None, uploaded_image=None)
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                img_pca = preprocess_image(filepath)
                
                if img_pca is not None:
                    prediction_encoded = gb_model.predict(img_pca)
                    prediction_proba = gb_model.predict_proba(img_pca)
                    confidence = np.max(prediction_proba) * 100
                    predicted_skin_type = label_encoder.inverse_transform(prediction_encoded)[0]
                    
                    recommendations = get_skin_type_recommendations(predicted_skin_type)
                    
                    conn = get_db_connection()
                    conn.execute(
                        'INSERT INTO predictions (user_id, image_path, predicted_skin_type, confidence_score) VALUES (?, ?, ?, ?)',
                        (session['user_id'], filename, predicted_skin_type, confidence)
                    )
                    conn.commit()
                    conn.close()
                    
                    prediction_result = {
                        'skin_type': predicted_skin_type,
                        'confidence': f"{confidence:.2f}%",
                        'recommendations': recommendations
                    }
                    uploaded_image = filename
                    
                    flash(f'✅ Your skin type is: {predicted_skin_type}', 'success')
                else:
                    flash('❌ Error processing image. Please try a different image.', 'danger')
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')
        else:
            flash('Invalid file type! Please upload PNG, JPG, or JPEG files.', 'danger')
    
    return render_template('prediction.html', 
                         prediction=prediction_result, 
                         uploaded_image=uploaded_image)

@app.route('/history')
@login_required
def history():
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)

@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM users WHERE id = ?', (session['user_id'],)
    ).fetchone()
    
    pred_count = conn.execute(
        'SELECT COUNT(*) as count FROM predictions WHERE user_id = ?',
        (session['user_id'],)
    ).fetchone()
    
    conn.close()
    
    return render_template('profile.html', user=user, pred_count=pred_count['count'])

@app.route('/condition-prediction', methods=['GET', 'POST'])
@login_required
def condition_prediction():
    """Skin condition prediction page"""
    global model_loaded
    
    prediction_result = None
    uploaded_image = None
    current_condition = None
    
    if not model_loaded:
        flash('⚠️ Model not loaded. Please check if model file exists.', 'danger')
        return render_template('condition_prediction.html', prediction=None, uploaded_image=None)
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                img_pca = preprocess_image(filepath)
                
                if img_pca is not None:
                    prediction_encoded = gb_model.predict(img_pca)
                    prediction_proba = gb_model.predict_proba(img_pca)
                    confidence = np.max(prediction_proba) * 100
                    predicted_condition = label_encoder.inverse_transform(prediction_encoded)[0]
                    
                    # Get condition-specific recommendations
                    recommendations = get_skin_condition_recommendations(predicted_condition)
                    
                    # Store in session
                    session['current_condition'] = predicted_condition
                    session['current_condition_recommendations'] = recommendations
                    
                    # Store in database
                    conn = get_db_connection()
                    conn.execute(
                        'INSERT INTO predictions (user_id, image_path, predicted_skin_type, confidence_score) VALUES (?, ?, ?, ?)',
                        (session['user_id'], filename, predicted_condition, confidence)
                    )
                    conn.commit()
                    conn.close()
                    
                    prediction_result = {
                        'condition': predicted_condition,
                        'confidence': f"{confidence:.2f}%",
                        'recommendations': recommendations
                    }
                    uploaded_image = filename
                    current_condition = predicted_condition
                    
                    flash(f'✅ Prediction complete! Detected: {predicted_condition}', 'success')
                else:
                    flash('❌ Error processing image. Please try again.', 'danger')
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')
        else:
            flash('Invalid file type!', 'danger')
    
    return render_template('condition_prediction.html', 
                         prediction=prediction_result, 
                         uploaded_image=uploaded_image,
                         current_condition=current_condition)

@app.route('/doctors-routine')
@login_required
def doctors_routine():
    """Display doctor's recommended routine page"""
    condition = session.get('current_condition')
    recommendations = session.get('current_condition_recommendations')
    
    if not condition or not recommendations:
        flash('Please upload an image first to see doctor\'s recommendations.', 'warning')
        return redirect(url_for('condition_prediction'))
    
    return render_template('doctors_routine.html', 
                         condition=condition, 
                         doctors_routine=recommendations['doctors_routine'],
                         description=recommendations['description'])

@app.route('/condition-products')
@login_required
def condition_products():
    """Display condition-specific products page"""
    condition = session.get('current_condition')
    recommendations = session.get('current_condition_recommendations')
    
    if not condition or not recommendations:
        flash('Please upload an image first to see product recommendations.', 'warning')
        return redirect(url_for('condition_prediction'))
    
    return render_template('condition_products.html', 
                         condition=condition, 
                         products=recommendations['products'],
                         description=recommendations['description'])

@app.route('/condition-routine')
@login_required
def condition_routine():
    """Display condition-specific skincare routine page"""
    condition = session.get('current_condition')
    recommendations = session.get('current_condition_recommendations')
    
    if not condition or not recommendations:
        flash('Please upload an image first to see skincare routine.', 'warning')
        return redirect(url_for('condition_prediction'))
    
    return render_template('condition_routine.html', 
                         condition=condition, 
                         routine=recommendations['routine'],
                         description=recommendations['description'])

@app.route('/food-intake')
@login_required
def food_intake():
    """Display food recommendations page"""
    condition = session.get('current_condition')
    recommendations = session.get('current_condition_recommendations')
    
    if not condition or not recommendations:
        flash('Please upload an image first to see food recommendations.', 'warning')
        return redirect(url_for('condition_prediction'))
    
    return render_template('food_intake.html', 
                         condition=condition, 
                         foods=recommendations['food_intake'],
                         description=recommendations['description'])

@app.route('/questionnaire', methods=['GET', 'POST'])
@login_required
def questionnaire():
    """Skin problem questionnaire page"""
    result = None
    
    if request.method == 'POST':
        # Get questionnaire data
        symptoms = request.form.get('symptoms', '')
        location = request.form.get('location', '')
        duration = request.form.get('duration', '')
        triggers = request.form.get('triggers', '')
        other_details = request.form.get('other_details', '')
        
        # Analyze the responses to determine possible skin condition
        possible_condition = analyze_questionnaire_responses(symptoms, location, duration, triggers, other_details)
        
        # Get detailed information about the condition
        condition_info = get_skin_condition_details(possible_condition)
        
        result = {
            'condition': possible_condition,
            'info': condition_info,
            'symptoms': symptoms,
            'location': location,
            'duration': duration,
            'triggers': triggers
        }
        
        flash(f'Based on your responses, possible condition: {possible_condition}', 'info')
    
    return render_template('questionnaire.html', result=result)

def analyze_questionnaire_responses(symptoms, location, duration, triggers, other_details):
    """Analyze questionnaire responses to determine possible facial skin condition"""
    symptoms_lower = symptoms.lower()
    location_lower = location.lower()
    triggers_lower = triggers.lower()
    details_lower = other_details.lower()
    
    # Facial acne patterns
    if any(word in symptoms_lower for word in ['pimple', 'acne', 'breakout', 'zit', 'blackhead', 'whitehead', 'cyst']):
        if any(word in location_lower for word in ['forehead', 't-zone', 'nose', 'chin', 'jaw']):
            if 'jaw' in location_lower or 'chin' in location_lower:
                return 'Hormonal Acne'
            elif 'forehead' in location_lower or 'nose' in location_lower:
                return 'Acne Vulgaris (T-zone)'
            else:
                return 'Acne Vulgaris'
    
    # Facial redness and rosacea patterns
    elif any(word in symptoms_lower for word in ['red', 'redness', 'flushing', 'blush']):
        if any(word in location_lower for word in ['cheek', 'nose', 'forehead', 'chin']):
            if any(word in triggers_lower for word in ['spicy', 'alcohol', 'sun', 'hot', 'stress']):
                return 'Rosacea'
            else:
                return 'Facial Redness / Rosacea'
    
    # Dryness and flaking patterns
    elif any(word in symptoms_lower for word in ['dry', 'scaly', 'flake', 'peeling', 'rough']):
        if any(word in location_lower for word in ['cheek', 'forehead', 'eyebrow', 'nose']):
            if 'eyebrow' in location_lower or 'nose' in location_lower:
                return 'Seborrheic Dermatitis'
            else:
                return 'Dry Facial Skin / Xerosis'
    
    # Dark spots and pigmentation
    elif any(word in symptoms_lower for word in ['dark', 'brown', 'patch', 'spot', 'pigmentation', 'melasma']):
        if any(word in location_lower for word in ['cheek', 'forehead', 'upper lip', 'nose']):
            return 'Melasma / Hyperpigmentation'
    
    # Itching and sensitivity
    elif any(word in symptoms_lower for word in ['itch', 'itching', 'burn', 'sting']):
        if any(word in location_lower for word in ['face', 'cheek', 'around eyes']):
            return 'Facial Eczema / Contact Dermatitis'
    
    # Wrinkles and aging
    elif any(word in symptoms_lower for word in ['wrinkle', 'fine line', 'aging', 'sagging']):
        return 'Aging Skin Concerns'
    
    # Oiliness
    elif any(word in symptoms_lower for word in ['oily', 'greasy', 'shine']):
        if any(word in location_lower for word in ['forehead', 'nose', 't-zone']):
            return 'Oily Skin / Seborrhea'
    
    # Default
    return 'General Facial Skin Concern'
def get_skin_condition_details(condition):
    """Get detailed information about a skin condition including brief and cure"""
    
    conditions_details = {
        'Eczema': {
            'brief': 'Eczema (atopic dermatitis) is a chronic inflammatory skin condition that causes dry, itchy, and inflamed patches of skin. It often appears in childhood and can persist into adulthood. The condition is linked to an overactive immune response to irritants or allergens.',
            'basic_cure': [
                '🩺 **Immediate Relief**: Apply cold compresses to reduce itching and inflammation',
                '💧 **Moisturize Frequently**: Use thick, fragrance-free moisturizers at least 2-3 times daily, especially after bathing',
                '🚿 **Gentle Cleansing**: Take short, lukewarm showers and use mild, soap-free cleansers',
                '🧴 **Topical Treatments**: Apply over-the-counter hydrocortisone cream (1%) for mild flare-ups',
                '🌡️ **Avoid Triggers**: Identify and avoid triggers like harsh soaps, wool fabrics, stress, and certain foods',
                '💊 **Antihistamines**: Oral antihistamines can help control itching, especially at night',
                '🌿 **Natural Soothers**: Oatmeal baths and aloe vera gel can provide soothing relief',
                '🏥 **Medical Care**: See a dermatologist if symptoms persist or worsen; prescription treatments may be needed'
            ],
            'home_remedies': [
                'Apply coconut oil or shea butter for natural moisturizing',
                'Use colloidal oatmeal in bath water to soothe irritated skin',
                'Apply aloe vera gel directly to affected areas',
                'Use apple cider vinegar diluted with water (1:1 ratio) as a compress',
                'Take warm baths with Epsom salts',
                'Apply honey to affected areas for its antibacterial properties'
            ],
            'prevention': [
                'Maintain a consistent moisturizing routine',
                'Use a humidifier in dry environments',
                'Wear soft, breathable cotton clothing',
                'Avoid scratching - keep nails short',
                'Manage stress through relaxation techniques',
                'Identify and avoid food triggers'
            ],
            'warning_signs': 'Seek medical attention if you experience: severe itching disrupting sleep, signs of infection (oozing, yellow crusts, fever), widespread rash, or if over-the-counter treatments aren\'t working.'
        },
        
        'Psoriasis': {
            'brief': 'Psoriasis is an autoimmune condition that causes rapid buildup of skin cells, resulting in thick, red, scaly patches that are often itchy and sometimes painful. It\'s a chronic condition with periods of flare-ups and remission.',
            'basic_cure': [
                '🧴 **Topical Treatments**: Use over-the-counter creams containing salicylic acid or coal tar to reduce scaling',
                '💧 **Moisturize**: Apply thick moisturizers daily to reduce dryness and scaling',
                '🌞 **Sunlight Exposure**: Moderate sun exposure can help improve symptoms (use sunscreen on unaffected areas)',
                '🛁 **Bath Therapy**: Take warm baths with Epsom salts or oatmeal to soothe skin and remove scales',
                '💊 **Anti-inflammatory**: Over-the-counter anti-inflammatory medications may help with pain and swelling',
                '🌿 **Natural Remedies**: Aloe vera, capsaicin cream, and tea tree oil may provide relief',
                '🍎 **Healthy Lifestyle**: Maintain a healthy weight, avoid alcohol, and quit smoking to reduce flare-ups',
                '🏥 **Medical Treatment**: See a dermatologist for prescription treatments including topical steroids, light therapy, or systemic medications'
            ],
            'home_remedies': [
                'Apply aloe vera gel directly to plaques several times daily',
                'Use capsaicin cream to reduce pain and inflammation',
                'Take oatmeal baths to soothe itching',
                'Apply apple cider vinegar to the scalp to relieve itching',
                'Use turmeric as an anti-inflammatory supplement',
                'Apply dead sea salts in bath water'
            ],
            'prevention': [
                'Manage stress through meditation and exercise',
                'Avoid skin injuries (cuts, scrapes, sunburns)',
                'Limit alcohol consumption',
                'Quit smoking',
                'Maintain a healthy weight',
                'Avoid cold, dry weather when possible'
            ],
            'warning_signs': 'Seek immediate medical care if you develop: pus-filled blisters, fever, severe joint pain (possible psoriatic arthritis), or if plaques cover large areas of your body.'
        },
        
        'Acne Vulgaris': {
            'brief': 'Acne vulgaris is a common skin condition that occurs when hair follicles become clogged with oil and dead skin cells. It causes whiteheads, blackheads, pimples, and sometimes deeper cysts or nodules. It most commonly affects adolescents but can occur at any age.',
            'basic_cure': [
                '🧼 **Gentle Cleansing**: Wash face twice daily with a mild, non-comedogenic cleanser',
                '🔬 **Benzoyl Peroxide**: Use OTC benzoyl peroxide products to kill bacteria and dry excess oil',
                '⚗️ **Salicylic Acid**: Apply products with salicylic acid to unclog pores',
                '💧 **Oil-Free Moisturizer**: Use lightweight, non-comedogenic moisturizers even if skin is oily',
                '🚫 **Don\'t Pick**: Never pop or squeeze pimples to prevent scarring and infection',
                '🧴 **Spot Treatments**: Apply targeted spot treatments containing sulfur or tea tree oil',
                '💊 **Oral Options**: Consider over-the-counter supplements like zinc or vitamin A',
                '🏥 **Professional Care**: See a dermatologist for persistent acne; prescription options include retinoids, antibiotics, or isotretinoin'
            ],
            'home_remedies': [
                'Apply ice to reduce inflammation of large pimples',
                'Use tea tree oil diluted with carrier oil as spot treatment',
                'Apply green tea compresses for antioxidant benefits',
                'Use honey as an antibacterial face mask',
                'Apply aloe vera to reduce redness',
                'Use apple cider vinegar as a toner (diluted)'
            ],
            'prevention': [
                'Keep hands away from face',
                'Change pillowcases weekly',
                'Use non-comedogenic makeup and skincare products',
                'Avoid heavy, oil-based hair products',
                'Remove makeup before sleeping',
                'Manage stress levels',
                'Avoid excessive sun exposure'
            ],
            'warning_signs': 'Seek medical attention if you have: severe, painful cysts or nodules, scarring, emotional distress from acne, or if over-the-counter treatments haven\'t improved your skin after 2-3 months.'
        },
        
        'Rosacea': {
            'brief': 'Rosacea is a chronic inflammatory skin condition that causes facial redness, visible blood vessels, and sometimes small, red, pus-filled bumps. It typically affects the central face (cheeks, nose, forehead) and can cause eye irritation. Flare-ups are often triggered by certain foods, weather, or stress.',
            'basic_cure': [
                '🧴 **Gentle Skincare**: Use mild, fragrance-free cleansers and moisturizers',
                '🌞 **Sun Protection**: Apply broad-spectrum sunscreen (SPF 30+) daily; physical blockers (zinc oxide) are best',
                '❄️ **Cool Compresses**: Apply cold compresses to reduce redness and inflammation',
                '🚫 **Avoid Triggers**: Identify and avoid personal triggers (spicy foods, alcohol, hot drinks, extreme temperatures)',
                '💊 **Topical Treatments**: Use OTC products with azelaic acid or metronidazole (prescription strength may be needed)',
                '🌿 **Soothing Ingredients**: Look for skincare with niacinamide, green tea, or licorice extract',
                '💄 **Camouflage**: Use green-tinted makeup to neutralize redness',
                '🏥 **Medical Care**: See a dermatologist for prescription medications, laser therapy, or other treatments'
            ],
            'home_remedies': [
                'Apply green tea compresses to reduce inflammation',
                'Use aloe vera gel for soothing relief',
                'Apply cucumber slices to calm irritated skin',
                'Use colloidal oatmeal masks',
                'Apply chamomile tea compresses',
                'Use honey masks for their anti-inflammatory properties'
            ],
            'prevention': [
                'Protect skin from sun exposure daily',
                'Avoid known triggers',
                'Manage stress through relaxation techniques',
                'Use gentle, non-abrasive skincare',
                'Avoid overheating during exercise',
                'Limit alcohol and spicy foods'
            ],
            'warning_signs': 'Seek medical attention if you experience: eye irritation or vision changes (ocular rosacea), thickening skin on nose (rhinophyma), or if symptoms are significantly affecting your quality of life.'
        },
        
        'Contact Dermatitis': {
            'brief': 'Contact dermatitis is a localized skin reaction caused by direct contact with an irritant or allergen. It appears as red, itchy, inflamed skin that may blister or ooze. Common triggers include poison ivy, nickel, fragrances, and certain chemicals.',
            'basic_cure': [
                '🧼 **Immediate Cleansing**: Wash affected area thoroughly with mild soap and water',
                '❄️ **Cold Compresses**: Apply cool, wet compresses to reduce inflammation and itching',
                '🧴 **Topical Treatments**: Use over-the-counter hydrocortisone cream (1%) for mild reactions',
                '💊 **Oral Antihistamines**: Take antihistamines like cetirizine or diphenhydramine to control itching',
                '🚫 **Avoid Irritant**: Identify and avoid contact with the triggering substance',
                '🌿 **Calamine Lotion**: Apply calamine lotion to soothe itching and dry oozing areas',
                '🛁 **Oatmeal Baths**: Take colloidal oatmeal baths to calm irritated skin',
                '🏥 **Medical Care**: See a doctor if the reaction is severe, covers large areas, or shows signs of infection'
            ],
            'home_remedies': [
                'Apply aloe vera gel for cooling relief',
                'Use baking soda paste (3:1 ratio with water) to reduce itching',
                'Apply apple cider vinegar diluted with water as a compress',
                'Use coconut oil to moisturize and protect skin',
                'Apply witch hazel to reduce inflammation',
                'Use cucumber slices for cooling effect'
            ],
            'prevention': [
                'Identify and avoid known allergens',
                'Use protective gloves when handling chemicals',
                'Patch test new skincare products',
                'Choose hypoallergenic, fragrance-free products',
                'Wear protective clothing when gardening or hiking',
                'Keep skin moisturized to maintain barrier function'
            ],
            'warning_signs': 'Seek immediate medical care if you experience: difficulty breathing, swelling of face or throat, severe blistering, signs of infection (pus, fever), or if the rash covers a large portion of your body.'
        },
        
        'General Skin Concern': {
            'brief': 'Based on your responses, we recommend general skin care practices while monitoring your symptoms. Skin concerns can vary widely, and maintaining good skin health is important for overall wellness.',
            'basic_cure': [
                '💧 **Hydration**: Drink plenty of water throughout the day',
                '🧴 **Moisturize Daily**: Use a gentle, fragrance-free moisturizer appropriate for your skin type',
                '🌞 **Sun Protection**: Apply broad-spectrum sunscreen (SPF 30+) every day',
                '🧼 **Gentle Cleansing**: Use mild, non-irritating cleansers twice daily',
                '🍎 **Balanced Diet**: Eat a diet rich in fruits, vegetables, and healthy fats',
                '😴 **Adequate Sleep**: Get 7-9 hours of quality sleep for skin repair',
                '🧘 **Stress Management**: Practice relaxation techniques to reduce stress-related skin issues',
                '🏥 **Professional Consultation**: If symptoms persist, consult a dermatologist for proper diagnosis'
            ],
            'home_remedies': [
                'Apply aloe vera for general skin soothing',
                'Use honey masks for antibacterial benefits',
                'Apply coconut oil for gentle moisturizing',
                'Take oatmeal baths for overall skin health',
                'Use green tea compresses for antioxidant benefits'
            ],
            'prevention': [
                'Establish a consistent skincare routine',
                'Protect skin from environmental damage',
                'Stay hydrated and eat nutritious foods',
                'Get regular exercise to improve circulation',
                'Avoid harsh chemicals and irritants',
                'Get adequate sleep and manage stress'
            ],
            'warning_signs': 'If you experience persistent or worsening symptoms, spreading rash, fever, severe pain, or signs of infection, please consult a healthcare provider promptly.'
            
        }
    }
    
    # Return details for the specific condition, or default if not found
    for key in conditions_details.keys():
        if key.lower() in condition.lower() or condition.lower() in key.lower():
            return conditions_details[key]
    
    return conditions_details['General Skin Concern']

@app.route('/check-model')
def check_model():
    global model_loaded, gb_model, label_encoder, pca
    
    if model_loaded:
        return jsonify({
            'loaded': True,
            'model_type': str(type(gb_model)),
            'classes': list(label_encoder.classes_),
            'pca_components': pca.n_components_
        })
    else:
        return jsonify({
            'loaded': False,
            'error': 'Model not loaded',
            'model_path_exists': os.path.exists(MODEL_PATH)
        })



@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    init_db()
    
    print("\n" + "="*50)
    print("STARTING FLASK APPLICATION")
    print("="*50)
    
    if load_model():
        print("\n✅ Model ready for predictions!")
    else:
        print("\n⚠️ WARNING: Model failed to load.")
        print(f"   Please ensure '{MODEL_PATH}' exists in: {os.getcwd()}")
    
    print("\n" + "="*50)
    print("SERVER INFORMATION")
    print("="*50)
    print("Access the application at: http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)