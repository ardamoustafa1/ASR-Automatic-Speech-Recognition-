import re

file_path = "ASR.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replacements to strip light theme colors and adapt to dark mode variables or just remove them
replacements = [
    ('style="color:#101828;background:#ffffff;"', 'class="empty-state"'),
    ('style="color:#2563eb;"', 'style="color:var(--asr-accent); font-weight:800; text-transform:uppercase; font-size:0.8rem; margin-bottom:0.5rem;"'),
    ('style="color:#101828;"', 'style="color:var(--asr-text); font-size:1.1rem; font-weight:600; margin-bottom:0.5rem;"'),
    ('style="color:#667085;"', 'style="color:var(--asr-muted); font-size:0.9rem;"'),
    ('style="color:#34d399"', 'style="color:var(--asr-success)"'),
    ('style="background:#0f1720;color:#d8e4f2;border:1px solid #243244;"', 'class="glass-card" style="margin-top:1rem;"'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Inline styles patched in ASR.py!")
