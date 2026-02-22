import streamlit as st
import plotly.express as px
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re

# ---------------- Load Sentence-BERT Model ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- Helper Functions ----------------
def extract_text_from_docx(docx_file):
    """Extract text from DOCX resume."""
    try:
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except:
        return ""

def get_embeddings(text_list):
    """Generate SBERT embeddings."""
    return model.encode(text_list)

def calculate_similarity(job_desc, resumes):
    """Calculate similarity between job description and resumes using SBERT embeddings."""
    job_embed = get_embeddings([job_desc])[0]
    resume_embeds = get_embeddings(resumes)
    return cosine_similarity([job_embed], resume_embeds)[0]

def extract_skills_from_jd(job_desc):
    """Simple skill extraction from job description using regex (keywords)."""
    common_skills = [
        "python", "java", "c++", "sql", "nlp", "machine learning", "deep learning",
        "tensorflow", "pytorch", "streamlit", "flask", "django", "excel", "powerbi",
        "data analysis", "cloud", "aws", "azure", "gcp", "hadoop", "spark", "tableau"
    ]
    jd_lower = job_desc.lower()
    extracted = [skill for skill in common_skills if re.search(rf"\b{re.escape(skill)}\b", jd_lower)]
    return list(set(extracted))  # unique skills

def analyze_skills(required_skills, resume_text):
    """Find matched and missing skills in the resume."""
    resume_lower = resume_text.lower()
    matched = [skill for skill in required_skills if re.search(rf"\b{re.escape(skill.lower())}\b", resume_lower)]
    missing = [skill for skill in required_skills if skill not in matched]
    return matched, missing

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="TalentMatch Resume Analyzer", page_icon="📄", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("📄 TalentMatch")
st.sidebar.markdown("### AI-powered Resume Scoring & Skill Gap Analysis")
st.sidebar.info("1. Enter job description\n2. Upload DOCX resumes\n3. Analyze results with SBERT\n4. View shortlist")

# ---------------- Main Title ----------------
st.title("✨ TalentMatch Resume Scoring & Fit Analyzer")
st.markdown("### Semantic Resume-Job Matching using Sentence-BERT (SBERT)")

# ---------------- Inputs ----------------
job_desc_input = st.text_area("Enter Job Description:")

resume_files = st.file_uploader("Upload Candidate Resumes (.docx)", type=["docx"], accept_multiple_files=True)

# ---------------- Start Analysis ----------------
if st.button("🚀 Start Analysis"):
    if not job_desc_input:
        st.warning("Please enter the Job Description first!")
    elif not resume_files:
        st.warning("Please upload at least one DOCX resume!")
    else:
        # Extract job description skills
        required_skills = extract_skills_from_jd(job_desc_input)

        # Process resumes
        resumes_text = [extract_text_from_docx(r) for r in resume_files]
        scores = calculate_similarity(job_desc_input, resumes_text)

        results_data = []

        # Tabs for results
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Detailed Analysis", "📈 Comparison Chart", "✅ Shortlisted Candidates", "⬇️ Download Reports"]
        )

        with tab1:
            st.subheader("📊 Resume-wise Analysis (Semantic Matching)")
            for i, resume in enumerate(resume_files):
                score = scores[i] * 100
                matched, missing = analyze_skills(required_skills, resumes_text[i])
                results_data.append({
                    "Resume": resume.name,
                    "Match %": round(score, 2),
                    "Matched Skills": ", ".join(matched) if matched else "None",
                    "Missing Skills": ", ".join(missing) if missing else "None"
                })

                with st.container():
                    st.markdown(f"### {resume.name}")
                    st.write(f"**Match Score:** {score:.2f}%")
                    st.write(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
                    st.write(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

                    # Pie Chart
                    labels = ['Match Score', 'Gap']
                    values = [score, 100 - score]
                    fig_pie = px.pie(
                        values=values,
                        names=labels,
                        title=f"Semantic Fit for {resume.name}",
                        color=labels,
                        color_discrete_map={'Match Score': '#2ecc71', 'Gap': '#e74c3c'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("📈 Resume Comparison Chart")
            fig_bar = px.bar(
                x=[r.name for r in resume_files],
                y=scores * 100,
                labels={'x': 'Resumes', 'y': 'Match Score (%)'},
                title="Resume vs Match Score",
                color=scores * 100,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("✅ Shortlisted Candidates (Score ≥ 70%)")
            df = pd.DataFrame(results_data).sort_values(by="Match %", ascending=False).reset_index(drop=True)
            shortlisted = df[df["Match %"] >= 70]

            if not shortlisted.empty:
                st.dataframe(shortlisted, use_container_width=True)
                st.success("Top Candidates Shortlisted based on Match Percentage!")
            else:
                st.warning("No candidates met the 70% cutoff for shortlisting.")

        with tab4:
            st.subheader("⬇️ Download Candidate Reports")
            df = pd.DataFrame(results_data).sort_values(by="Match %", ascending=False).reset_index(drop=True)

            # Download full results as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv,
                file_name="resume_analysis_results.csv",
                mime="text/csv",
            )

else:
    st.info("Enter a job description, upload DOCX resumes, then click **Start Analysis** to see results.")
