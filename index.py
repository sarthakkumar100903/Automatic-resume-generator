import os
import subprocess
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Create a .env file with GOOGLE_API_KEY="your_key_here"

# --- 1. HARDCODED BASE RESUME CONTENT ---
# ... (Your BASE_EXPERIENCE_TEX, BASE_PROJECTS_TEX, and BASE_SKILLS_TEX variables remain the same) ...
BASE_EXPERIENCE_TEX = r"""
\resumeSubHeadingListStart
    \resumeSubheading
      {Member Of Technical Staff Intern}{January 2025 – June 2025}
      {\textbf{Nutanix}}{Bengaluru, India}
      \resumeItemListStart
        \item \textbf{Tech Stack:} Python, React, Redux, Go-Lang, Calm, Epsilon
        \item Led development of a PoC Kubernetes service provider on Prism Central, enabling Helm chart deployment across 5+ multi-cluster environments, reducing manual provisioning overhead.
        \item Designed and implemented Nested Runbook support, enabling Blueprints to invoke other Runbooks; improved workflow modularity by 40% and reduced logic duplication by 30% across Calm workflows, enhancing maintainability and scalability.
        \item Integrated Macros, Dynamic, and Global Variables into the React UI and backend to standardize configuration, reducing manual input errors and setup time by 25%.
      \resumeItemListEnd
    \resumeSubheading
      {Data Scientist Intern}{May 2024 – August 2024}
      {\textbf{Teliolabs Communication Private limited}}{Hyderabad, India}
      \resumeItemListStart
        \item \textbf{Tech Stack:} Python, NLTK, Faiss DB, Bert, Tensorflow, React, Django.
        \item Formulated an ML model to predict solutions for Jira Tickets by processing and summarizing ticket data from Confluence, storing embeddings in vector DB; achieved 85% accuracy in solution recommendations, reducing manual resolution time by roughly 40%.
      \resumeItemListEnd
\resumeSubHeadingListEnd
"""

BASE_PROJECTS_TEX = r"""
\resumeSubHeadingListStart
    \resumeProject
      {{\textbf{Runalytix – Intelligent Runbook Validator and Optimizer}} \href{https://youtu.be/aLlhrwAPgxA}{\textit{\small{Video}}} \textbar{} \href{https://docs.google.com/presentation/d/1qt7XzuZkbnZ_yUeG8MJ2ZQZqVL7QH2_8/edit?slide=id.p1#slide=id.p1}{\textit{\small{Design Doc}}}}
      {}
      {May 2025}
      {}
      \resumeItemListStart
        \item \textbf{Tools \& Technologies:} Python, FastAPI, React, Redux, LangChain, RAG, OpenAI, Nutanix Calm, Epsilon, IDF
        \item Devised \textbf{Runalytix}, an AI assistant for validating and optimizing \textbf{Calm Runbook scripts} in real-time, reducing debugging time by \textbf{~50%}.
        \item Implemented \textbf{LLM-based analysis} to detect syntax, logic, and security issues in Shell, PowerShell, and HTTP tasks, achieving \textbf{~90\% issue detection accuracy}.
        \item Integrated a context-aware suggestion engine using \textbf{LangChain + RAG}, improving fix recommendation accuracy by \textbf{35\%} and reducing Runbook execution errors by \textbf{~40\%}.
      \resumeItemListEnd

    \resumeProject
      {{\textbf{Sustainable Traffic Management system}} \href{https://github.com/shubhamvermaa/Hack_36?tab=readme-ov-file}{\textit{\small{Github}}} \textbar{} \href{https://youtu.be/P73OS4Aaz14}{\textit{\small{Video}}}}
      {}
      {Jan. – Apr. 2024}
      {}
      \resumeItemListStart
        \item \textbf{Tools \& Technologies:} JavaScript, Python, HTML, CSS, Gsheet, SheetDB, P5.js, YoloV4
        \item Used \textbf{Braess's Paradox} to optimize traffic flow, increasing simulated throughput by \textbf{30\%} in dense areas.
        \item Developed a \textbf{custom algorithm} to model congestion across \textbf{50+ city layouts}, analyzing flow and bottlenecks.
        \item Designed a \textbf{city planning tool} to simulate and evaluate the impact of new road construction in urban layouts, improving traffic efficiency predictions by \textbf{~35\%} and aiding data-driven infrastructure decisions.
        \item Leveraged \textbf{YOLOv4} for traffic detection from video and trained a routing model to optimize path selection, improving speed by \textbf{25\%}.
      \resumeItemListEnd

    \resumeProject
      {{\textbf{Intelligent Chatbot for User Queries}} \href{https://github.com/dubeykirtiman/MNNIT_ChatBot}{\textit{\small{Github}}} \textbar{} \href{https://www.youtube.com/watch?v=mfgmaNysABo}{\textit{\small{Video}}}}
      {}
      {Sep. - Dec. 2023}
      {}
      \resumeItemListStart
        \item \textbf{Tools \& Technologies:} RNN, LSTM, Python, OpenAI (v3.5), NLTK, RASA, Spacy, MySQL, Ngrok, AWS EC2
        \item Built a context-aware chatbot for \textbf{MNNIT} and \textbf{Botrush} using intent recognition and dialogue management, achieving \textbf{75\%+ intent classification accuracy}.
        \item Embedded \textbf{OpenAI API (v3.5)} for generate fallback responses when user queries fell outside predefined intents.
        \item Deployed the bot on \textbf{Telegram}, handling \textbf{100+ user queries} via \textbf{AWS EC2 t2.micro} using \textbf{Ngrok} tunneling.
      \resumeItemListEnd
\resumeSubHeadingListEnd
"""

BASE_SKILLS_TEX = r"""
\section{\textbf{Technical Skills}}
  \resumeHeadingSkillStart
    \resumeSkillListStart
      \resumeItem{\textbf{Programming Languages}}{C, C++, Java, Python, JavaScript}
      \resumeItem{\textbf{Web Development}}{HTML, CSS, MERN Stack (MySQL, Express.js, React, Node.js)}
      \resumeItem{\textbf{Frameworks and Libraries}}{TensorFlow, Keras, OpenCV, NumPy, Pandas, NLTK}
      \resumeItem{\textbf{Databases}}{MySQL, Mongo Db, Cassandra DB, Faiss DB}
      \resumeItem{\textbf{Generative AI and ML}}{LangChain, OpenAI API, BERT, LLAMA 3, and ML/DL concepts}
      \resumeItem{\textbf{Areas of Interest}}{Backend Development, AI/ML, Distributed Systems, Data Structures and Algorithms}
    \resumeSkillListEnd
  \resumeHeadingSkillEnd
"""

# --- File Paths ---
JOB_DESC_FILE = 'job_description.txt'
USER_CONTEXT_FILE = 'user_context.txt'
EXPERIENCE_OUTPUT_FILE = 'sections/experience.tex'
PROJECTS_OUTPUT_FILE = 'sections/projects.tex'
SKILLS_OUTPUT_FILE = 'sections/skills.tex'
MASTER_LATEX_FILE = 'resume_master.tex'

def compile_pdf():
    """Compiles the LaTeX document to a PDF."""
    print(f"Compiling {MASTER_LATEX_FILE} to PDF...")
    for i in range(2):
        process = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', MASTER_LATEX_FILE],
            capture_output=True, text=True
        )
        if process.returncode != 0:
            print(f"--- LaTeX Compilation Error (Pass {i+1}) ---")
            print(process.stdout)
            return False
    pdf_output_name = MASTER_LATEX_FILE.replace('.tex', '.pdf')
    print(f"PDF compilation successful! Output saved to {pdf_output_name}")
    return True

def main():
    """Main function to run the resume generation process."""
    print("--- Starting Automated Resume Generation (Hardcoded Edition) ---")

    # Initialize LLM and Prompt
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Using the more powerful model
    output_parser = StrOutputParser()
    
    # IMPROVED PROMPT
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert resume writer. Your task is to significantly rewrite and tailor the Experience, Projects, and Skills sections of a LaTeX resume to perfectly match a job description.

        **HIGHEST PRIORITY INSTRUCTIONS FROM USER:**
        ---
        {user_context}
        ---

        **TARGET JOB DESCRIPTION:**
        ---
        {job_description}
        ---

        **BASE LaTeX CONTENT FOR EXPERIENCE:**
        ---
        {base_experience}
        ---

        **BASE LaTeX CONTENT FOR PROJECTS:**
        ---
        {base_projects}
        ---

        **BASE LaTeX CONTENT FOR SKILLS:**
        ---
        {base_skills}
        ---

        **YOUR TASK:**
        You MUST rewrite, rephrase, and reorder the content in the base LaTeX sections to align with the Job Description and User Context. Do not simply return the original text.
        1.  In the Experience and Projects sections, you MUST rephrase the bullet points to use keywords and reflect the responsibilities mentioned in the job description. Emphasize achievements and quantifiable results that are most relevant to the role.
        2.  In the Skills section, you MUST re-organize and prioritize the skills to highlight what's most important for the job.
        3.  Crucially, you must maintain the exact original LaTeX structure and commands. Do not change the LaTeX formatting.

        **OUTPUT FORMAT:**
        Provide the full, rewritten LaTeX code for all three sections, enclosed in the following tags. Do not add any other commentary.

        <EXPERIENCE_TEX_START>
        (The complete, rewritten content for experience.tex goes here)
        <EXPERIENCE_TEX_END>

        <PROJECTS_TEX_START>
        (The complete, rewritten content for projects.tex goes here)
        <PROJECTS_TEX_END>

        <SKILLS_TEX_START>
        (The complete, rewritten content for skills.tex goes here)
        <SKILLS_TEX_END>
        """
    )

    # Create the LangChain Chain
    chain = prompt_template | llm | output_parser

    # Read the job description and user context from files
    print("Reading job description and user context...")
    with open(JOB_DESC_FILE, 'r', encoding='utf-8') as f: job_desc = f.read()
    with open(USER_CONTEXT_FILE, 'r', encoding='utf-8') as f: user_context = f.read()

    # Invoke the chain with the hardcoded content
    print("Calling the LLM via LangChain to tailor the resume...")
    llm_response = chain.invoke({
        "user_context": user_context,
        "job_description": job_desc,
        "base_experience": BASE_EXPERIENCE_TEX,
        "base_projects": BASE_PROJECTS_TEX,
        "base_skills": BASE_SKILLS_TEX
    })
    print("LLM response received.")

    # Parse the response and overwrite the files
    try:
        new_experience = re.search(r'<EXPERIENCE_TEX_START>(.*?)<EXPERIENCE_TEX_END>', llm_response, re.DOTALL).group(1).strip()
        new_projects = re.search(r'<PROJECTS_TEX_START>(.*?)<PROJECTS_TEX_END>', llm_response, re.DOTALL).group(1).strip()
        new_skills = re.search(r'<SKILLS_TEX_START>(.*?)<SKILLS_TEX_END>', llm_response, re.DOTALL).group(1).strip()
        
        print(f"\n--- Overwriting {EXPERIENCE_OUTPUT_FILE} ---\n{new_experience}\n")
        with open(EXPERIENCE_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(new_experience)

        print(f"--- Overwriting {PROJECTS_OUTPUT_FILE} ---\n{new_projects}\n")
        with open(PROJECTS_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(new_projects)

        print(f"--- Overwriting {SKILLS_OUTPUT_FILE} ---\n{new_skills}\n")
        with open(SKILLS_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(new_skills)

        print("All .tex files updated successfully.")

        # Compile the final PDF
        #compile_pdf()

    except AttributeError:
        print("\n--- ERROR ---")
        print("Could not parse the LLM response. The model may not have followed the format instructions.")
        print("LLM Output:\n", llm_response)

    print("--- Process Finished ---")

if __name__ == "__main__":
    main()