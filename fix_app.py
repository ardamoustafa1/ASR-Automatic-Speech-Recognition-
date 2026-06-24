with open("ASR/ASR.py", "r") as f:
    content = f.read()
content = content.replace("render_app()", """
try:
    st.write("DEBUG: STARTING APP")
    render_app()
    st.write("DEBUG: RENDER APP DONE")
except Exception as e:
    st.error(f"CRITICAL ERROR: {e}")
    import traceback
    st.code(traceback.format_exc())
""")
with open("ASR/ASR.py", "w") as f:
    f.write(content)
