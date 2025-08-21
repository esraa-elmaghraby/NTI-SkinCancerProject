from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io, base64, os, uuid, logging, hashlib
from datetime import datetime

# ============ إعدادات عامة ============
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/resnet_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ أسماء الكلاسات ============
CLASS_NAMES = {
    0: 'akiec', 1: 'bcc', 2: 'bkl',
    3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
}

CLASS_DESCRIPTIONS = {
    'akiec': {"full_name": "Actinic keratoses and intraepithelial carcinoma",
              "description": "Pre-cancerous lesion caused by sun damage.",
              "risk": "High", "color": "orange"},
    'bcc': {"full_name": "Basal cell carcinoma",
            "description": "Most common form of skin cancer, requires treatment.",
            "risk": "High", "color": "orange"},
    'bkl': {"full_name": "Benign keratosis-like lesions",
            "description": "Generally harmless, monitor for changes.",
            "risk": "Very Low", "color": "green"},
    'df': {"full_name": "Dermatofibroma",
           "description": "Benign fibrous nodule, no treatment usually required.",
           "risk": "Very Low", "color": "green"},
    'mel': {"full_name": "Melanoma",
            "description": "Aggressive cancer, urgent treatment required.",
            "risk": "Very High", "color": "red"},
    'nv': {"full_name": "Melanocytic nevi (Common mole)",
           "description": "Common benign mole, generally harmless.",
           "risk": "Very Low", "color": "green"},
    'vasc': {"full_name": "Vascular lesions",
             "description": "Usually harmless vascular malformations.",
             "risk": "Low", "color": "blue"},
}

# ============ Preprocessing ============
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============ تحميل الموديل ============
model = None
def load_model():
    global model
    try:
        logger.info(f"🔄 Loading model from {MODEL_PATH}")
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 7)

        if not os.path.exists(MODEL_PATH):
            logger.error("❌ Model file not found")
            return False

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

# ============ Utilities ============
def preprocess_image(image_data):
    try:
        if isinstance(image_data, str) and "data:image" in image_data:
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return val_transform(image).unsqueeze(0), image_bytes
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {e}")
        return None, None

# ============ كاش للتنبؤات ============
PREDICTION_CACHE = {}

def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def predict_skin_lesion(image_bytes, image_tensor):
    try:
        # لو الصورة موجودة في الكاش نرجع نفس النتيجة
        img_hash = get_image_hash(image_bytes)
        if img_hash in PREDICTION_CACHE:
            return PREDICTION_CACHE[img_hash]

        # Prediction من الموديل
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            result = (int(predicted_class.item()), float(confidence.item()))
            # نخزن في الكاش
            PREDICTION_CACHE[img_hash] = result
            return result
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None, None

# ============ Routes ============
@app.route("/")
def home():
    return render_template("index.html")   # HTML لازم يكون في مجلد templates/

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # بيانات الفورم
        gender = request.form.get("gender", "").strip()
        age = request.form.get("age", "").strip()
        location = request.form.get("location", "unknown").strip()
        file = request.files.get("image")

        if not gender or not age or not file:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        try:
            age = int(age)
            if age < 1 or age > 120:
                return jsonify({"success": False, "error": "Invalid age"}), 400
        except:
            return jsonify({"success": False, "error": "Age must be a number"}), 400

        # قراءة الصورة
        image_data = file.read()
        image_tensor, image_bytes = preprocess_image(image_data)
        if image_tensor is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        # Prediction
        pred_idx, confidence = predict_skin_lesion(image_bytes, image_tensor)
        if pred_idx is None:
            return jsonify({"success": False, "error": "Prediction failed"}), 500

        class_key = CLASS_NAMES[pred_idx]
        class_info = CLASS_DESCRIPTIONS[class_key]

        response = {
            "success": True,
            "analysis_id": f"DS-{datetime.now().year}-{str(uuid.uuid4())[:8].upper()}",
            "prediction": {
                "type": class_info["full_name"],
                "class_key": class_key,
                "confidence": f"{confidence*100:.1f}%",
                "risk": class_info["risk"],
                "color": class_info["color"],
                "description": class_info["description"]
            },
            "patient_info": {
                "age": age,
                "gender": gender,
                "location": location,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "model_info": {
                "model_type": "ResNet-50",
                "classes": len(CLASS_NAMES),
                "device": str(DEVICE)
            }
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Error in /analyze: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat()
    })

# ============ Main ============
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
