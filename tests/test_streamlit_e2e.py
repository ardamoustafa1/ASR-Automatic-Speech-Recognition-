from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads():
    at = AppTest.from_file("tools/legacy_streamlit/ASR/ASR.py")
    # Setting an environment variable so it doesn't crash on default API URLs
    at.run()
    assert not at.exception
    # App should render its main components (e.g. radio buttons)
    assert len(at.radio) > 0
    assert "Seçiniz" in at.radio[0].label
