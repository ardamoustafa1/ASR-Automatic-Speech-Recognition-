from streamlit.testing.v1 import AppTest

at = AppTest.from_file("tools/legacy_streamlit/ASR/ASR.py")
at.run()
print("Exception:", at.exception)
print("Title:", at.title)
print("Sidebar:", len(at.sidebar))
print("Radio:", len(at.radio))
if len(at.radio) > 0:
    print("Radio labels:", [r.label for r in at.radio])
