import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
LOW_CONFIDENCE = 0.70

# Display names and colors for the API's class keys
CLASSES = {
    "bact": {"name": "Bacterial keratitis", "color": "#c2410c"},
    "fung": {"name": "Fungal keratitis", "color": "#7c3aed"},
    "healthy": {"name": "Healthy", "color": "#15803d"},
}

st.set_page_config(page_title="Keratitis Screening", page_icon="👁️", layout="wide")


@st.cache_data(ttl=30)
def check_health():
    """Ping the FastAPI backend; cached so it doesn't run on every rerun."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return True, r.json().get("device", "?")
    except requests.RequestException:
        return False, None


def predict(uploaded_file):
    """Send the image to /predict the same way the HTML page's FormData does."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    r = requests.post(f"{API_URL}/predict", files=files, timeout=30)
    if not r.ok:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def probability_bar(name, p, color):
    st.markdown(
        f"""
        <div style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;font-size:0.9rem">
            <span>{name}</span><span>{p * 100:.1f}%</span>
          </div>
          <div style="height:10px;background:#e6ebf1;border-radius:5px;overflow:hidden">
            <div style="width:{p * 100}%;height:100%;background:{color}"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Header
head_left, head_right = st.columns([4, 1])
with head_left:
    st.title("Keratitis Screening Assistant")
    st.caption("Bacterial · Fungal · Healthy - ResNet50 classifier")
with head_right:
    online, device = check_health()
    if online:
        st.success(f"Service online · {device}")
    else:
        st.error("Service unavailable")

left, right = st.columns(2, gap="large")

# Upload panel
with left:
    st.subheader("Patient image")
    uploaded = st.file_uploader(
        "Slit-lamp / clinical eye photograph", type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        st.image(uploaded, use_container_width=True)
    analyze = st.button(
        "Analyze image", type="primary", disabled=not (uploaded and online), use_container_width=True
    )

# Result panel
with right:
    st.subheader("Analysis result")
    if not uploaded:
        st.info("Upload an image and click **Analyze** to see the model's assessment.")
    elif analyze:
        try:
            with st.spinner("Analyzing…"):
                data = predict(uploaded)
        except (requests.RequestException, RuntimeError) as e:
            st.error(f"Analysis failed: {e}")
        else:
            top = CLASSES.get(data["predicted_class"], {"name": data["predicted_class"], "color": "#0b6e8a"})
            st.markdown(
                f"""
                <div style="border-left:5px solid {top['color']};padding:12px 16px;
                            background:#f3f6f9;border-radius:6px;margin-bottom:18px">
                  <div style="font-size:0.8rem;color:#5f7184">Most likely finding</div>
                  <div style="font-size:1.5rem;font-weight:700;color:{top['color']}">{top['name']}</div>
                  <div style="font-size:0.9rem;color:#5f7184">Model confidence: {data['confidence'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Class probabilities**")
            for key, p in sorted(data["probabilities"].items(), key=lambda kv: kv[1], reverse=True):
                c = CLASSES.get(key, {"name": key, "color": "#0b6e8a"})
                probability_bar(c["name"], p, c["color"])

            if data["confidence"] < LOW_CONFIDENCE:
                st.warning(
                    "**Low confidence.** The model is uncertain between classes. "
                    "Treat this result as inconclusive and refer for clinical evaluation."
                )

st.divider()
st.caption(
    "For research and educational use only. This tool is not a medical device and must not be used as the "
    "sole basis for diagnosis or treatment. Bacterial keratitis recall on the held-out test set was 0.50, so "
    "negative or low-confidence results need clinical confirmation (e.g. corneal scraping and culture)."
)
