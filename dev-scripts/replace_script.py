
asr_file = "ASR.py"
css_file = "new_css.css"

with open(asr_file, encoding="utf-8") as f:
    lines = f.readlines()

with open(css_file, encoding="utf-8") as f:
    new_css = f.read()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "def local_css():" in line:
        start_idx = i
        break

if start_idx != -1:
    for i in range(start_idx + 1, len(lines)):
        if '""", unsafe_allow_html=True)' in line or '    """, unsafe_allow_html=True)' in lines[i]:
            end_idx = i
            break

if start_idx != -1 and end_idx != -1:
    # Replace the block
    new_lines = lines[:start_idx + 1]
    new_lines.append('    st.markdown("""\n')
    new_lines.append(new_css + '\n')
    new_lines.append('    """, unsafe_allow_html=True)\n')
    new_lines.extend(lines[end_idx + 1:])

    with open(asr_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("CSS successfully replaced!")
else:
    print(f"Could not find local_css block boundaries. Start: {start_idx}, End: {end_idx}")
