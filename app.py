import streamlit as st
import pandas as pd
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Analyzer")
st.write("Upload your resume and check your job compatibility.")


# -----------------------------
# Skills Database
# -----------------------------
SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "nlp",
    "power bi",
    "tableau",
    "excel",
    "aws",
    "azure",
    "docker",
    "git",
    "github",
    "html",
    "css",
    "javascript",
    "react",
    "flask",
    "fastapi"
]


# -----------------------------
# Extract Text From PDF
# -----------------------------
def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Extract Skills
# -----------------------------
def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill.lower() in text:
            found_skills.append(skill)

    return sorted(set(found_skills))


# -----------------------------
# Calculate Similarity
# -----------------------------
def calculate_similarity(resume_text, job_text):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        [resume_text, job_text]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(similarity * 100, 2)


# -----------------------------
# Job Dataset
# -----------------------------
jobs = {

    "Data Analyst": """
    Python SQL Excel Power BI Tableau
    data analysis pandas numpy statistics
    """,

    "Machine Learning Engineer": """
    Python machine learning deep learning
    pandas numpy scikit-learn tensorflow
    data science artificial intelligence
    """,

    "AI Engineer": """
    Python artificial intelligence machine learning
    deep learning NLP tensorflow pytorch
    """,

    "Web Developer": """
    HTML CSS JavaScript React Python
    Flask FastAPI Git GitHub
    """,

    "Cloud Engineer": """
    Python AWS Azure Docker Git
    cloud computing deployment
    """
}


# -----------------------------
# Resume Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your Resume PDF",
    type=["pdf"]
)


if uploaded_file:

    with st.spinner("Analyzing resume..."):

        resume_text = extract_pdf_text(
            uploaded_file
        )

        if not resume_text.strip():

            st.error(
                "Resume से text नहीं मिल पाया। "
                "कृपया text-based PDF upload करें।"
            )

        else:

            # Extract skills
            resume_skills = extract_skills(
                resume_text
            )

            st.success(
                "Resume successfully analyzed!"
            )

            # -------------------------
            # Resume Information
            # -------------------------

            st.subheader("🧠 Detected Skills")

            if resume_skills:

                st.write(
                    ", ".join(
                        skill.title()
                        for skill in resume_skills
                    )
                )

            else:

                st.warning(
                    "कोई predefined skill नहीं मिली।"
                )


            # -------------------------
            # Job Matching
            # -------------------------

            st.subheader(
                "🎯 Job Recommendations"
            )

            results = []

            for job_name, job_description in jobs.items():

                score = calculate_similarity(
                    resume_text,
                    job_description
                )

                job_skills = extract_skills(
                    job_description
                )

                missing_skills = [
                    skill
                    for skill in job_skills
                    if skill not in resume_skills
                ]

                results.append({

                    "Job": job_name,

                    "Match": score,

                    "Missing Skills":
                        ", ".join(missing_skills)

                })


            # Sort jobs
            results = sorted(
                results,
                key=lambda x: x["Match"],
                reverse=True
            )


            # -------------------------
            # Display Results
            # -------------------------

            for result in results:

                col1, col2 = st.columns(
                    [3, 1]
                )

                with col1:

                    st.write(
                        f"### 💼 {result['Job']}"
                    )

                    if result[
                        "Missing Skills"
                    ]:

                        st.write(
                            "Missing Skills: "
                            + result[
                                "Missing Skills"
                            ]
                        )

                    else:

                        st.write(
                            "Missing Skills: None"
                        )

                with col2:

                    st.metric(
                        "Match",
                        f"{result['Match']}%"
                    )

                st.divider()


            # -------------------------
            # Top Recommendation
            # -------------------------

            best_job = results[0]

            st.subheader(
                "⭐ Recommended Job"
            )

            st.success(
                f"{best_job['Job']} "
                f"({best_job['Match']}% Match)"
            )


            # -------------------------
            # Resume Preview
            # -------------------------

            with st.expander(
                "📄 View Extracted Resume Text"
            ):

                st.text(
                    resume_text[:10000]
                )