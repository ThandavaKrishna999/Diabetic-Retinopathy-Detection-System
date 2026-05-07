from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(filename, text, color, folder='static/uploads'):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    
    # Create image
    img = Image.new('RGB', (800, 800), color=color)
    d = ImageDraw.Draw(img)
    
    # Draw text (centered)
    # Default font
    try:
        # Try to use a nice font if available, else default
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()
        
    # Draw text
    d.text((50, 350), text, fill=(255, 255, 255), font=font)
    d.text((50, 450), "Placeholder Image", fill=(255, 255, 255), font=font)
    
    img.save(path)
    print(f"Created {path}")

# Files referenced in seed_data.py
files = [
    ('demo_mild_left.jpg', 'Mild DR - Left Eye\n(6 Months Ago)', '#2ecc71'),
    ('demo_mod_left.jpg', 'Moderate DR - Left Eye\n(3 Months Ago)', '#f39c12'),
    ('demo_severe_left.jpg', 'Severe DR - Left Eye\n(Today)', '#e74c3c')
]

# Heatmaps
heatmaps = [
    ('demo_heatmap_1.jpg', 'Heatmap - Mild', '#27ae60'),
    ('demo_heatmap_2.jpg', 'Heatmap - Moderate', '#d35400'),
    ('demo_heatmap_3.jpg', 'Heatmap - Severe', '#c0392b')
]

for f, t, c in files:
    create_placeholder(f, t, c, 'static/uploads')

for f, t, c in heatmaps:
    create_placeholder(f, t, c, 'static/heatmaps')

print("All placeholder images generated successfully.")
