
with open('ASR/ui_components.py') as f:
    lines = f.readlines()

new_lines = lines[:1130]
new_lines.append("def render_app():\n")

for line in lines[1130:]:
    if line == '\n':
        new_lines.append(line)
    else:
        new_lines.append("    " + line)

with open('ASR/ui_components.py', 'w') as f:
    f.writelines(new_lines)

with open('ASR/ASR.py', 'a') as f:
    f.write("\nrender_app()\n")

print("Done")
