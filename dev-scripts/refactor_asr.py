
def refactor():
    with open('ASR/ASR.py', encoding='utf-8') as f:
        lines = f.readlines()

    # Find boundaries based on function names
    logic_start = -1
    ui_start = -1

    for i, line in enumerate(lines):
        if line.startswith('def get_ffmpeg_path():') and logic_start == -1:
            logic_start = i
        if line.startswith('def display_detection_results(') and ui_start == -1:
            ui_start = i

    if logic_start == -1 or ui_start == -1:
        print("Could not find boundaries")
        return

    # Extract sections
    imports_and_classes = lines[:logic_start]
    logic_section = lines[logic_start:ui_start]

    # We also need to split out the Streamlit main code which starts at:
    # "st.set_page_config" or similar, wait let's find main logic
    main_start = -1
    for i in range(ui_start, len(lines)):
        if "st.set_page_config" in lines[i]:
            main_start = i
            break

    if main_start == -1:
        ui_section = lines[ui_start:]
        main_section = []
    else:
        ui_section = lines[ui_start:main_start]
        main_section = lines[main_start:]

    # Write logic_handlers.py
    with open('ASR/logic_handlers.py', 'w', encoding='utf-8') as f:
        # Need imports for logic
        f.write("import os, re, math, time, subprocess, shutil, hashlib, wave\n")
        f.write("from pathlib import Path\n")
        f.write("from difflib import SequenceMatcher\n")
        f.write("import streamlit as st\n")
        f.write("from collections import namedtuple\n")
        f.write("from datetime import timedelta\n")
        f.write("from array import array\n")
        f.write("import srt\n")
        f.write("from ASR.config import *\n")
        f.write("from ASR.ASR import *\n") # Circular import risk, let's just write the lines
        f.write("".join(logic_section))

    # Write ui_components.py
    with open('ASR/ui_components.py', 'w', encoding='utf-8') as f:
        f.write("import streamlit as st\n")
        f.write("import base64, re, json, time\n")
        f.write("from ASR.config import *\n")
        f.write("from ASR.logic_handlers import *\n")
        f.write("".join(ui_section))

    # Rewrite ASR.py
    with open('ASR/ASR.py', 'w', encoding='utf-8') as f:
        f.write("".join(imports_and_classes))
        f.write("\nfrom ASR.logic_handlers import *\n")
        f.write("from ASR.ui_components import *\n\n")
        f.write("".join(main_section))

if __name__ == '__main__':
    refactor()
