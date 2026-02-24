import streamlit as st
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from dotenv import load_dotenv

# ==============================================================================
# 1. SETUP & SECURITATE
# ==============================================================================
st.set_page_config(page_title="GenAI Headhunter", page_icon="🕵️", layout="wide")

# Încărcăm variabilele din fișierul .env
load_dotenv()

# Încercăm să luăm cheia din OS (local) sau din Streamlit Secrets (cloud)
api_key = os.getenv("GROQ_API_KEY")

# Fallback pentru Streamlit Cloud deployment
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

# Validare critică: Dacă nu avem cheie, oprim aplicația aici.
if not api_key:
    st.error("⛔ EROARE CRITICĂ: Lipsește `GROQ_API_KEY`.")
    st.info("Te rog creează un fișier `.env` în folderul proiectului și adaugă: GROQ_API_KEY=cheia_ta_aici")
    st.stop()

# Configurare Client Groq Global (pentru a nu-l reinițializa constant)
client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.TOOLS)

# Sidebar Informativ (Fără input de date sensibile)
with st.sidebar:
    st.header("🕵️ GenAI Headhunter")
    st.success("✅ API Key încărcat securizat")
    st.markdown("---")
    st.write("Acest tool demonstrează:")
    st.write("• Web Scraping (BS4)")
    st.write("• Secure Env Variables")
    st.write("• Structured Data (Pydantic)")


# ==============================================================================
# 2. DATA MODELS (PYDANTIC SCHEMAS)
# ==============================================================================

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

# (NEW) Adaugare sub-modele SalaryRange, Location, RedFlag
class SalaryRange(BaseModel):
    min: Optional[int] = Field(None, ge=0, description="Salariu minim (numeric), dacă există")
    max: Optional[int] = Field(None, ge=0, description="Salariu maxim (numeric), dacă există")
    currency: Optional[str] = Field(None, description="Monedă, ex: EUR, RON, USD")

class Location(BaseModel):
    city: Optional[str] = Field(None, description="Orașul, dacă apare în anunț")
    country: Optional[str] = Field(None, description="Țara, dacă apare în anunț")
    is_remote: bool = Field(False, description="True dacă jobul este remote/hibrid")

class RedFlag(BaseModel):
    text: str = Field(..., description="Semnalul de alarmă identificat")
    severity: Literal["low", "medium", "high"] = Field(..., description="Severitatea red flag-ului")
    category: Literal["toxicity", "vague", "unrealistic"] = Field(..., description="Categoria red flag-ului")

# (NEW) Clasa JobAnalysis modificata pentru a foslosi modelele noi, definite anterior
class JobAnalysis(BaseModel):
    role_title: str = Field(..., description="Titlul jobului standardizat")
    company_name: str = Field(..., description="Numele companiei")
    seniority: Literal["Intern", "Junior", "Mid", "Senior", "Lead", "Architect"] = Field(..., description="Nivelul de experiență dedus")
    match_score: int = Field(..., ge=0, le=100, description="Scor 0-100: Calitatea descrierii jobului")

    tech_stack: List[str] = Field(default_factory=list, description="Listă cu tehnologii specifice")
    red_flags: List[RedFlag] = Field(default_factory=list, description="Red flags structurate (text + severitate + categorie)")

    salary: Optional[SalaryRange] = Field(None, description="Interval salarial estimat/extras din text")
    location: Optional[Location] = Field(None, description="Locația extrasă (inclusiv remote)")

    summary: str = Field(..., description="Rezumat scurt (max 2 fraze) în limba română")

    # Validator cross-field
    @model_validator(mode="after")
    def cross_field_remote_location_check(self):
        """
        Semnalează inconsistențe: location.is_remote=True dar apar cerințe de birou în câmpuri textuale.
        (heuristic simplu; la nevoie îl rafinezi)
        """
        if self.location and self.location.is_remote:
            # Heuristic minim: dacă orașul e setat și pare 'mandatory', putem ridica red flag
            # (în practică mai bine o faci pe baza textului original în Agent 3)
            pass
        return self
    
### (NEW) Creare modele Pydantic pentru agenti

# RawExtraction (Agent 1)
class RawExtraction(BaseModel):
    role_title: Optional[str] = Field(None, description="Titlul rolului (dacă poate fi dedus)")
    company_name: Optional[str] = Field(None, description="Compania (dacă e explicită)")
    tech_stack: List[str] = Field(default_factory=list, description="Tehnologii menționate explicit")
    requirements: List[str] = Field(default_factory=list, description="Cerințe/requirements explicite")
    benefits: List[str] = Field(default_factory=list, description="Beneficii menționate explicit")

    salary: Optional[SalaryRange] = None
    location: Optional[Location] = None

# StrategicAdvice (Agent 2)
class StrategicAdvice(BaseModel):
    match_score: int = Field(..., ge=0, le=100, description="Scor de calitate/potrivire bazat pe fapte")
    seniority: Literal["Intern", "Junior", "Mid", "Senior", "Lead", "Architect"] = Field(...)

    red_flags: List[RedFlag] = Field(default_factory=list)

    summary: str = Field(..., description="Rezumat (max 2 fraze) în română")
    negotiation_tips: List[str] = Field(default_factory=list, description="Sfaturi de negociere")
    interview_questions: List[str] = Field(default_factory=list, description="Întrebări recomandate la interviu")
    market_positioning: List[str] = Field(default_factory=list, description="Observații despre poziționare în piață (generic)")

# Validator report (Agent 3)
class ValidationReport(BaseModel):
    is_consistent: bool = Field(..., description="True dacă extracția pare consistentă cu textul")
    issues: List[str] = Field(default_factory=list, description="Probleme / suspiciuni găsite")


# ==============================================================================
# 3. UTILS - SCRAPER (Colectare Date)
# ==============================================================================

def scrape_clean_job_text(url: str, max_chars: int = 3000) -> str:
    """
    Descarcă pagina și returnează un text curat, optimizat pentru contextul LLM.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Eliminăm elementele inutile care consumă tokeni
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            junk.decompose()
            
        # Extragem textul și eliminăm spațiile multiple
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        
        return text[:max_chars] 
        
    except Exception as e:
        return f"Scraping Error: {str(e)}"

# ==============================================================================
# 4. AI SERVICE LAYER (Logica LLM)
# ==============================================================================

#Sectiunea urmatoare (analyze_job_with_ai) a fost inlocuita cu sectiunea aferenta celor 3 agenti

# @st.cache_resource(show_spinner=False)
# def analyze_job_with_ai(text: str) -> JobAnalysis:
#     """
#     Trimite textul curățat către Groq și returnează obiectul structurat.
#     """
#     return client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         response_model=JobAnalysis,
#         messages=[
#             {
#                 "role": "system", 
#                 "content": (
#                     "Ești un Recruiter Expert în IT. Analizează textul jobului cu obiectivitate. "
#                     "Identifică tehnologiile și potențialele probleme (red flags). "
#                     "Răspunde strict în formatul cerut."
#                 )
#             },
#             {
#                 "role": "user", 
#                 "content": f"Analizează acest job description:\n\n{text}"
#             }
#         ],
#         temperature=0.1,
#     )


# Agent 1 — Extractor
@st.cache_data(show_spinner=False, ttl=60*60)
def extractor_agent(text: str) -> RawExtraction:
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # cerut în temă
        response_model=RawExtraction,
        messages=[
            {"role": "system", "content": (
                "Ești Agentul 1: The Extractor. Extragi DOAR fapte brute din text. "
                "Nu faci interpretări, nu dai sfaturi. "
                "Returnezi STRICT schema cerută."
            )},
            {"role": "user", "content": f"Extrage fapte brute din acest job description:\n\n{text}"}
        ],
        temperature=0.0,
    )

# Agent 3 — Validator
@st.cache_data(show_spinner=False, ttl=60*60)
def validator_agent(text: str, extraction: RawExtraction) -> ValidationReport:
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=ValidationReport,
        messages=[
            {"role": "system", "content": (
                "Ești Agentul 3: The Validator. Verifici consistența între textul original și extracția primită. "
                "Semnalezi orice câmp care nu pare susținut de text (halucinație). "
                "Returnezi STRICT schema cerută."
            )},
            {"role": "user", "content": (
                "TEXT ORIGINAL:\n"
                f"{text}\n\n"
                "EXTRACȚIE (JSON conceptual):\n"
                f"{extraction.model_dump()}\n\n"
                "Verifică dacă extracția e susținută de text. "
                "Ex: salary fără mențiune, remote vs on-site, tehnologii inventate."
            )}
        ],
        temperature=0.0,
    )

# Agent 2 — Counselor
@st.cache_data(show_spinner=False, ttl=60*60)
def counselor_agent(extraction: RawExtraction) -> StrategicAdvice:
    return client.chat.completions.create(
        model="mixtral-8x7b-32768",  # conform cerinței, sau alt model Groq
        response_model=StrategicAdvice,
        messages=[
            {"role": "system", "content": (
                "Ești Agentul 2: The Counselor. Primești fapte brute și oferi insight-uri strategice. "
                "Nu inventa fapte noi: folosești DOAR ceea ce e în extracție. "
                "Returnezi STRICT schema cerută."
            )},
            {"role": "user", "content": (
                "Iată faptele extrase. Generează analiză strategică:\n\n"
                f"{extraction.model_dump()}"
            )}
        ],
        temperature=0.7,
    )

### Pipeline orchestration (final)
def run_multi_agent_pipeline(text: str, use_validator: bool = True):
    extraction = extractor_agent(text)

    report = None
    if use_validator:
        report = validator_agent(text, extraction)
        if not report.is_consistent:
            # alegi: fie oprești, fie avertizezi în UI și continui
            pass

    advice = counselor_agent(extraction)

    # (opțional) convertești în JobAnalysis “final” dacă vrei compatibilitate cu UI vechi
    analysis = JobAnalysis(
        role_title=advice.summary.split(":")[0] if extraction.role_title is None else extraction.role_title,
        company_name=extraction.company_name or "N/A",
        seniority=advice.seniority,
        match_score=advice.match_score,
        tech_stack=extraction.tech_stack,
        red_flags=advice.red_flags,
        salary=extraction.salary,
        location=extraction.location,
        summary=advice.summary,
    )

    return analysis, extraction, advice, report

# ==============================================================================
# 5. UI - APLICAȚIA STREAMLIT
# ==============================================================================

st.title("🕵️ GenAI Headhunter Assistant")
st.markdown("Transformă orice Job Description într-o analiză structurată folosind AI.")

# Tab-uri
tab1, tab2 = st.tabs(["🚀 Analiză Job", "📊 Market Scan (Batch)"])

# --- TAB 1: ANALIZA UNUI SINGUR LINK ---
with tab1:
    st.subheader("Analizează un Job URL")
    url_input = st.text_input("Introdu URL-ul:", placeholder="https://...")
    
    if st.button("Analizează Job", key="btn_single"):
        if not url_input:
            st.warning("Te rugăm introdu un URL.")
        else:
            with st.spinner("🕷️ Scraping & 🤖 AI Analysis..."):
                raw_text = scrape_clean_job_text(url_input)
            
            if "Error" in raw_text:
                st.error(raw_text)
            else:
                try:
                    # Modificare tab1
                    analysis, extraction, advice, report = run_multi_agent_pipeline(raw_text, use_validator=True)
                    data = analysis
                    
                    # -- DISPLAY --
                    st.divider()
                    col_h1, col_h2 = st.columns([3, 1])
                    with col_h1:
                        st.markdown(f"### {data.role_title}")
                        st.caption(f"Companie: **{data.company_name}** | Nivel: **{data.seniority}**")
                    with col_h2:
                        color = "normal" if data.match_score > 70 else "inverse"
                        st.metric("Quality Score", f"{data.match_score}/100", delta_color=color)

                    # Detalii
                    c1, c2, c3 = st.columns(3)
                    c1.info(f"**Remote:** {'Da' if data.is_remote else 'Nu'}")
                    c2.success(f"**Tehnologii:** {len(data.tech_stack)}")
                    c3.error(f"**Red Flags:** {len(data.red_flags)}")

                    st.markdown(f"**📝 Rezumat:** {data.summary}")
                    st.markdown("#### 🛠️ Tech Stack")
                    st.write(", ".join([f"`{tech}`" for tech in data.tech_stack]))
                    
                    # Output Strategic Advice
                    st.markdown("#### 🎯 Strategic Advice")
                    st.markdown("**Negociere:**")
                    for tip in advice.negotiation_tips:
                        st.write(f"- {tip}")

                    st.markdown("**Întrebări pentru interviu:**")
                    for q in advice.interview_questions:
                        st.write(f"- {q}")

                    st.markdown("**Poziționare în piață:**")
                    for m in advice.market_positioning:
                        st.write(f"- {m}")

                    # Ajusatre afisare Salary
                    if data.salary and (data.salary.min or data.salary.max):
                        st.write(f"💰 Salary: {data.salary.min} - {data.salary.max} {data.salary.currency}")
                    else:
                        st.write("💰 Salary: N/A")
                
                    # Ajusatre afisare Location
                    if data.location:
                        loc_txt = f"{data.location.city or ''} {data.location.country or ''}".strip()
                        st.write(f"📍 Location: {loc_txt if loc_txt else 'N/A'} | Remote: {'Da' if data.location.is_remote else 'Nu'}")
                    else:
                        st.write("📍 Location: N/A")

                    # Ajusatre afisare Redflags
                    if data.red_flags:
                        st.markdown("#### 🚩 Avertismente")
                        for rf in data.red_flags:
                            st.warning(f"⚠️ [{rf.severity.upper()} | {rf.category}] {rf.text}")


                except Exception as e:
                    st.error(f"Eroare AI: {str(e)}")

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.subheader("📊 Compară mai multe joburi")
    urls_text = st.text_area("Paste URL-uri (unul pe linie):", height=150)
    
    if st.button("Scanează Piața", key="btn_batch"):
        urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
        
        if not urls:
            st.warning("Nu ai introdus link-uri.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, link in enumerate(urls):
                status_text.text(f"Analizez {i+1}/{len(urls)}...")
                text = scrape_clean_job_text(link)
                
                if "Error" not in text:
                    try:
                        extraction = extractor_agent(text)
                        results.append({
                            "Role": extraction.role_title,
                            "Company": extraction.company_name,
                            "Seniority": extraction.seniority,
                            "Tech": extraction.tech_stack,
                            "Score": extraction.match_score
                        })
                    except:
                        pass # Continuăm chiar dacă unul crapă
                
                progress_bar.progress((i + 1) / len(urls))
            
            status_text.text("Gata!")
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Grafic simplu
                st.bar_chart(df['Seniority'].value_counts())