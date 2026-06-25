import os

from PIL import Image, ImageDraw, ImageFont

os.makedirs('docs/assets', exist_ok=True)

frames = []
for i in range(30):
    img = Image.new('RGB', (800, 600), color=(15, 23, 42)) # Slate 900
    d = ImageDraw.Draw(img)

    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    text = "ASR Pro Enterprise\n\nFeature Showcase\n(Replace with actual demo recording)"

    # Simple animation: move a bar
    d.text((200, 250), text, fill=(255, 255, 255), align="center")
    d.rectangle([(100, 500), (100 + i * 20, 520)], fill=(59, 130, 246)) # Blue 500

    frames.append(img)

frames[0].save('docs/assets/demo.gif', format='GIF', append_images=frames[1:], save_all=True, duration=1000, loop=0)
print("demo.gif generated successfully.")
