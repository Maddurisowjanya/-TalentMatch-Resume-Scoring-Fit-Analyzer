import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def get_embeddings(text_list):
    """Generate embeddings for given texts using SBERT."""
    return model.encode(text_list)

def calculate_similarity(job_desc, resumes):
    """Calculate cosine similarity between job description and resumes."""
    job_embed = get_embeddings([job_desc])[0]
    resume_embeds = get_embeddings(resumes)
    scores = cosine_similarity([job_embed], resume_embeds)[0]
    return scores

def analyze_skills(job_text, resume_text):
    """Find matched and missing skills based on word overlap."""
    job_skills = set(job_text.lower().split())
    resume_skills = set(resume_text.lower().split())
    matched = job_skills & resume_skills
    missing = job_skills - resume_skills
    return matched, missing
def analyze_skills(job_text, resume_text):
    job_skills = set(job_text.lower().split())
    resume_skills = set(resume_text.lower().split())
    matched = job_skills & resume_skills
    missing = job_skills - resume_skills
    return matched, missing
