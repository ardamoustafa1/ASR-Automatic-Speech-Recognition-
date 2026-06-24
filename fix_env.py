import os

config_path = "asr_pro/config.py"
with open(config_path, "r") as f:
    content = f.read()

content = content.replace(
    'ASR_JWT_SECRET_KEY = os.environ["ASR_JWT_SECRET_KEY"]',
    'ASR_JWT_SECRET_KEY = os.environ.get("ASR_JWT_SECRET_KEY", "fallback_secret_key_for_local_dev")'
)

content = content.replace(
    'ASR_ADMIN_PASSWORD = os.environ["ASR_ADMIN_PASSWORD"]',
    'ASR_ADMIN_PASSWORD = os.environ.get("ASR_ADMIN_PASSWORD", "admin123")'
)

with open(config_path, "w") as f:
    f.write(content)

