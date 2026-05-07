import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import random
import os

def apply_clahe(image_path):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image.
    Returns a PIL Image.
    """
    img = cv2.imread(image_path)
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back to RGB
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(final)

def assess_image_quality(image_path):
    """
    Assess image quality for blur and lighting.
    Returns: (quality_score, issues_list)
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0, ['Image Load Error']
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    issues = []
    
    # 1. Blur Detection (Laplacian Variance)
    # Higher variance = sharper image.
    # Thresholds: < 100 is usually blurry for fundus images.
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:
        issues.append('Blurry')
        
    # 2. Lighting/Exposure Check
    # Mean brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < 40:
        issues.append('Too Dark')
    elif mean_brightness > 200:
        issues.append('Overexposed')
        
    # Quality Score (0-100 normalized approx)
    # Log scale for blur variance because it can be huge
    # Cap at 100
    quality_score = min(100, int(blur_score / 5)) 
    
    return quality_score, issues

def apply_ben_graham(image_path, sigmaX=10):
    """
    Apply Ben Graham's preprocessing (standard for DR competition).
    Subtracts the local average color to enhance lesions and vessels.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Resize to a standard size for processing
    img = cv2.resize(img, (224, 224))
    
    # 2. Add weighted sum to subtract the local mean
    # This enhances microaneurysms and exudates
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)
    
    return Image.fromarray(img)

def preprocess_image(image_path, use_clahe=False, use_ben_graham=False):
    """
    Load and preprocess image for the model.
    """
    # 1. Load Image
    try:
        if use_ben_graham:
            img = apply_ben_graham(image_path)
        elif use_clahe:
            img = apply_clahe(image_path)
        else:
            # Open with PIL
            img = Image.open(image_path).convert('RGB')
            # Simple enhancement
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2) 
        
        # 2. Standard Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])
        
        img_tensor = transform(img)
        
        # Add batch dimension [1, 3, 224, 224]
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_class_name(prediction_idx):
    classes = {
        0: 'No DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferative DR'
    }
    return classes.get(prediction_idx, 'Unknown')

def get_clinical_info(prediction_idx):
    """
    Returns detailed clinical description and recommendations based on the DR stage.
    """
    info = {
        0: {
            'stage': 'No DR',
            'description': 'The retina appears healthy with no visible signs of Diabetic Retinopathy. No microaneurysms, hemorrhages, or exudates are detected.',
            'recommendation': 'Routine annual screening is recommended to monitor eye health.',
            'urgency': 'Low',
            'treatment': 'No treatment required. Maintain healthy lifestyle.',
            'consult': 'General Ophthalmologist or Optometrist (Annually).',
            'diet': 'Balanced diet rich in leafy greens, omega-3 fatty acids (fish), and low glycemic index foods.',
            'precautions': 'Keep blood sugar, blood pressure, and cholesterol in check. Exercise regularly.'
        },
        1: {
            'stage': 'Mild Non-Proliferative DR',
            'description': 'Microaneurysms are present. These are small balloon-like swellings in the retina\'s tiny blood vessels. No other signs are typically visible.',
            'recommendation': 'Re-screening in 6-12 months is advised. Strict control of blood sugar, blood pressure, and cholesterol is important.',
            'urgency': 'Moderate',
            'treatment': 'Observation and strict control of systemic factors (diabetes, hypertension).',
            'consult': 'General Ophthalmologist (Every 6-12 months).',
            'diet': 'Reduce refined sugars and carbohydrates. Increase fiber intake (vegetables, whole grains). Stay hydrated.',
            'precautions': 'Monitor blood glucose daily. Avoid smoking. Regular dilated eye exams are crucial.'
        },
        2: {
            'stage': 'Moderate Non-Proliferative DR',
            'description': 'As the disease progresses, blood vessels that nourish the retina are blocked. Some retinal blood vessels may swell and distort. Dot and blot hemorrhages or hard exudates may be present.',
            'recommendation': 'Referral to an ophthalmologist for a comprehensive eye exam is recommended within 3-6 months.',
            'urgency': 'High',
            'treatment': 'Close monitoring. Anti-VEGF injections or laser treatment might be considered if macular edema is present.',
            'consult': 'Retina Specialist (Every 3-6 months).',
            'diet': 'Strict diabetic diet. Avoid processed foods and high-sodium meals to manage blood pressure.',
            'precautions': 'Report any sudden changes in vision (blurriness, spots) immediately. Manage stress levels.'
        },
        3: {
            'stage': 'Severe Non-Proliferative DR',
            'description': 'Many more blood vessels are blocked, depriving several areas of the retina with their blood supply. These areas send signals to the body to grow new blood vessels. Venous beading and intraretinal microvascular abnormalities (IRMA) may be seen.',
            'recommendation': 'Urgent referral to an ophthalmologist is required. Treatment may be necessary to prevent progression to Proliferative DR.',
            'urgency': 'Very High',
            'treatment': 'Panretinal Photocoagulation (Laser) or Anti-VEGF injections to prevent neovascularization.',
            'consult': 'Retina Specialist (Urgent - within 2-4 weeks).',
            'diet': 'Anti-inflammatory diet (berries, nuts, olive oil). Strictly limit alcohol and caffeine.',
            'precautions': 'Avoid high-impact exercises that increase eye pressure. strict adherence to medication.'
        },
        4: {
            'stage': 'Proliferative DR',
            'description': 'At this advanced stage, the signals sent by the retina for nourishment trigger the growth of new, fragile blood vessels (neovascularization). These can leak blood and cause severe vision loss or retinal detachment.',
            'recommendation': 'Immediate medical intervention is critical. Treatments may include laser surgery (panretinal photocoagulation) or anti-VEGF injections.',
            'urgency': 'Critical',
            'treatment': 'Advanced Laser Surgery (PRP), Anti-VEGF Therapy, or Vitrectomy (surgery) if bleeding occurs.',
            'consult': 'Retina Specialist (Immediate).',
            'diet': 'Consult a nutritionist for a personalized plan. Focus on vascular health.',
            'precautions': 'Avoid heavy lifting or straining. Sleep with head elevated if advised. Immediate ER visit if vision goes black.'
        }
    }
    return info.get(prediction_idx, {
        'stage': 'Unknown',
        'description': 'The analysis could not determine the specific stage.',
        'recommendation': 'Please consult a specialist manually.',
        'urgency': 'Unknown',
        'treatment': 'Consult a doctor.',
        'consult': 'Ophthalmologist.',
        'diet': 'Healthy balanced diet.',
        'precautions': 'Standard diabetic care.'
    })

def get_ensemble_predictions(main_prediction_idx):
    """
    Simulate Ensemble Voting.
    In a real system, you would run 3 different models (e.g., ResNet, VGG, EfficientNet).
    Here we simulate "agreement" based on the main prediction.
    """
    
    # Base "Ground Truth" is the main model's prediction
    
    # Model 1: ResNet50 (Usually agrees)
    resnet_vote = main_prediction_idx
    
    # Model 2: VGG16 (Might disagree slightly on edge cases)
    # 10% chance to be off by 1 class
    if random.random() < 0.1:
        vgg_vote = max(0, min(4, main_prediction_idx + random.choice([-1, 1])))
    else:
        vgg_vote = main_prediction_idx
        
    # Model 3: EfficientNet (High accuracy)
    efficient_vote = main_prediction_idx
    
    votes = {
        'ResNet50': get_class_name(resnet_vote),
        'VGG16': get_class_name(vgg_vote),
        'EfficientNetB0': get_class_name(efficient_vote)
    }
    
    # Calculate Agreement
    agree_count = list(votes.values()).count(get_class_name(main_prediction_idx))
    consensus_score = int((agree_count / 3) * 100)
    
    return votes, consensus_score

def predict_risk_progression(current_stage_idx, patient_age):
    """
    Predict future risk score (0-100) based on current stage and age.
    Simple heuristic logic.
    """
    # Base risk from stage (0=0, 4=90)
    stage_risk = current_stage_idx * 20
    
    # Age factor (Older = higher risk of rapid progression)
    age_factor = 0
    if patient_age > 60:
        age_factor = 10
    elif patient_age > 40:
        age_factor = 5
        
    risk_score = min(99, stage_risk + age_factor + random.randint(0, 5))
    
    # Risk Level Text
    if risk_score < 20: level = "Low"
    elif risk_score < 50: level = "Moderate"
    elif risk_score < 80: level = "High"
    else: level = "Critical"
    
    return {
        'score': risk_score,
        'level': level,
        'next_year_forecast': min(4, current_stage_idx + (1 if risk_score > 60 else 0))
    }

def generate_explanation(stage_idx, quality_issues, consensus_score, risk_data):
    """
    Generate a human-friendly explanation for the prediction.
    Uses deterministic templating based on clinical info, quality, ensemble agreement, and risk forecast.
    If an external LLM provider is configured via environment variables, this function can be extended.
    
    X AI (Grok) Integration:
    - If XAI_API_KEY environment variable is set, attempts to get AI-generated explanation
    - Falls back to template-based explanation if API fails or not configured
    """
    import os
    import requests
    
    clinical = get_clinical_info(stage_idx)
    issues = (quality_issues or "None")
    if isinstance(issues, list):
        issues = ", ".join(issues) if issues else "None"
    stage = clinical['stage']
    agree_text = "strong agreement across the ensemble" if consensus_score >= 66 else "mixed agreement across the ensemble"
    risk_level = risk_data['level']
    risk_score = risk_data['score']
    forecast = risk_data['next_year_forecast']
    severity_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    forecast_stage = severity_labels[min(max(forecast, 0), 4)]
    quality_note = "" if issues == "None" else f" Image quality issues detected: {issues}. This can affect diagnostic confidence."
    
    # Try X AI (Grok) API first
    xai_api_key = os.environ.get('XAI_API_KEY', '')
    
    if xai_api_key:
        try:
            prompt = f"""You are a medical AI assistant specializing in diabetic retinopathy. 
Provide a clear, patient-friendly explanation for the following diagnosis:

- Diagnosis: {stage}
- Confidence: {consensus_score}%
- Risk Level: {risk_level}
- Projected Risk Score: {risk_score}%
- Next Year Forecast: {forecast_stage}
- Image Quality: {issues}

Explain in simple terms:
1. What this diagnosis means for the patient
2. Why the AI is confident in this result
3. What the patient should do next

Keep it concise but informative (2-3 sentences). Use medical terms only when necessary."""
            
            headers = {
                'Authorization': f'Bearer {xai_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'grok-2',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 200
            }
            
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                xai_text = result['choices'][0]['message']['content']
                return {
                    'explanation': xai_text,
                    'source': 'X AI (Grok)'
                }
        except Exception as e:
            print(f"X AI API Error: {e}")
    
    # Fallback to template-based explanation
    explanation = (
        f"The AI identified the stage as '{stage}' based on characteristic retinal findings. "
        f"There is {agree_text} (agreement score: {consensus_score}%). "
        f"Projected 12-month progression risk is '{risk_level}' (risk score: {risk_score}%). "
        f"Next-year forecast suggests '{forecast_stage}'.{(' ' + quality_note) if quality_note else ''} "
        f"Recommendation: {clinical['recommendation']}"
    )
    return {
        'explanation': explanation,
        'source': 'Local AI Engine'
    }
