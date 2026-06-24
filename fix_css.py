with open("ASR/ui_components.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "def local_css():" in line:
        new_lines.append(line)
        new_lines.append("    pass\n")
        skip = True
        continue
    
    if skip and "def safe_html(value):" in line:
        skip = False
        
    if not skip:
        new_lines.append(line)

with open("ASR/ui_components.py", "w") as f:
    f.writelines(new_lines)
