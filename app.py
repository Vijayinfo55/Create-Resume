import base64
import mimetypes

import streamlit as st
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# ========= PAGE HEADER ==========
st.set_page_config(page_title="AI Resume Maker", page_icon="📄", layout="centered")
st.title("AI RESUME MAKER")
st.markdown("""USER CAN CREATE OR DOWNLOAD AI
CREATED RESUME BASED ON HIGH
ATS SCORE""")

# ========= API KEY LOAD ==========
GEMINI_API_KEY = st.sidebar.text_input("GEMINI_API_KEY", type="password")
GROQ_API_KEY = st.sidebar.text_input("GROQ_API_KEY", type="password")
TAVILY_API_KEY = st.sidebar.text_input("TAVILY_API_KEY", type="password")

# Don't touch the network / build the model until keys exist.
if not GEMINI_API_KEY:
    st.info("Enter your GEMINI_API_KEY in the sidebar to continue.")
    st.stop()


# ========= TOOL ==========
def search_recent_news_jobs(query: str):
    """Search recent news or jobs related to the given query
    (e.g. 'Python Developer jobs') and return relevant links."""
    if not TAVILY_API_KEY:
        return "TAVILY_API_KEY not provided, skipping web search."
    client = TavilyClient(api_key=TAVILY_API_KEY)
    return client.search(query)


# ========= MODEL + AGENT ==========
# NOTE: gemini-2.0-flash was retired by Google on June 1, 2026 (calls to it
# now fail with a 404/"model not found" style error). gemini-3.6-flash is
# the current generally-available (GA) Flash model as of July 2026. If this
# ever errors again, check the live list at
# https://ai.google.dev/gemini-api/docs/models and swap the string below.
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY,
)

agent = create_agent(
    model=model,
    tools=[search_recent_news_jobs],
)


def extract_text(content) -> str:
    """Safely pull plain text out of a LangChain message's .content,
    whether it's a plain string or a list of content blocks/dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


# ========= PHOTO HELPERS ==========
# We never send the raw image bytes to the LLM (wastes tokens and the model
# can't reliably echo back a huge base64 blob without corrupting it).
# Instead we ask the model to drop a placeholder <img> tag into the HTML,
# then swap the placeholder for the real base64 data-URI ourselves.
PHOTO_PLACEHOLDER = "{{RESUME_PHOTO_SRC}}"


def photo_to_data_uri(uploaded_file) -> str:
    """Convert a Streamlit UploadedFile (image) into a base64 data URI."""
    raw_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0]
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def inject_photo(html_code: str, photo_data_uri: str | None) -> str:
    """Replace the photo placeholder with the real image, or strip the
    placeholder tag out entirely if no photo was uploaded."""
    if photo_data_uri:
        return html_code.replace(PHOTO_PLACEHOLDER, photo_data_uri)
    # No photo provided: remove any <img ... PHOTO_PLACEHOLDER ... > tag the
    # model may have inserted so we don't end up with a broken image icon.
    import re

    pattern = r"<img[^>]*" + re.escape(PHOTO_PLACEHOLDER) + r"[^>]*>"
    return re.sub(pattern, "", html_code)


# ========= BASE PERSONA PROMPT ==========
BASE_PROMPT = """You are a helpful AI assistant and expert job resume/CV designer.
Your task is to produce a single, self-contained HTML-format resume with
modern, attractive CSS (and a little JS if it genuinely improves the look,
e.g. subtle hover effects) for a professional, polished result.

Design rules:
- Use a different styling and color palette each time (don't repeat the same
  theme run to run) -- e.g. rotate between minimalist, modern-sidebar,
  timeline-style, elegant-serif, bold-gradient-header, etc.
- Use a clean layout: a strong header/hero area with name and title, clear
  section headings (Summary, Experience, Education, Skills, Projects,
  Certifications, Contact, etc. -- only include sections supported by the
  user's details), good whitespace, readable fonts (system fonts or a
  Google Font import), and a cohesive color palette.
- Make it print-friendly (looks good if exported to PDF) as well as
  screen-friendly.
- Do not add any preamble, commentary, or explanation before or after the
  resume (no "here is your resume" type text) -- return only the HTML
  resume itself, starting with <!DOCTYPE html> or <html>.
- The resume should be tailored to the user's background (student or
  experienced professional) based on the details they provide, and written
  to maximize ATS (Applicant Tracking System) compatibility -- use standard
  section headings and real text (not text baked into images), even though
  the visual design is rich.

Photo rules:
- If the user has provided a photo, include EXACTLY ONE <img> tag somewhere
  tasteful in the header (e.g. a circular profile photo next to the name),
  with src="{PLACEHOLDER}" (use that exact literal string as the src -- do
  not invent or substitute any other URL). Style it nicely: circular or
  rounded-square, appropriate size (e.g. 120-160px), a subtle border or
  shadow that matches the palette, and object-fit: cover so it isn't
  distorted.
- If the user has NOT provided a photo, do not include any <img> tag at all.
""".format(PLACEHOLDER=PHOTO_PLACEHOLDER)


def build_prompt(user_details: str, has_photo: bool) -> str:
    photo_note = (
        "The user HAS uploaded a profile photo -- include the photo <img> tag "
        "as instructed above."
        if has_photo
        else "The user has NOT uploaded a photo -- do not include an <img> tag."
    )
    return f"{BASE_PROMPT}\n\n{photo_note}\n\nUser details:\n{user_details}"


# ========= UI ==========
photo_file = st.file_uploader(
    "Upload your photo (optional)", type=["png", "jpg", "jpeg", "webp"]
)
photo_data_uri = None
if photo_file is not None:
    photo_data_uri = photo_to_data_uri(photo_file)
    st.image(photo_file, caption="Preview", width=150)

details = st.text_area("Enter your details (education, experience, skills, etc.):")

if st.button("Generate Resume"):
    if not details.strip():
        st.warning("Please enter your details first.")
    else:
        with st.spinner("Running agent..."):
            query = build_prompt(details, has_photo=photo_data_uri is not None)
            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )
                last_message = response["messages"][-1]
                code = extract_text(last_message.content)
                code = inject_photo(code, photo_data_uri)
                st.components.v1.html(code, height=800, scrolling=True)
                st.download_button(
                    "Download Resume (HTML)",
                    data=code,
                    file_name="resume.html",
                    mime="text/html",
                )
            except Exception as e:
                st.error(f"Something went wrong while generating the resume: {e}")
