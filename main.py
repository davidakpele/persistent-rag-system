import streamlit as st
import os
import requests
import subprocess
import time
from pathlib import Path
import PyPDF2
import pdfplumber
import io
import docx
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import torch
import GPUtil
import json
import uuid
from datetime import datetime

# --- Config ---
OLLAMA_HOST = "http://localhost:11434"
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploaded_docs"
UPLOAD_DIR.mkdir(exist_ok=True)
HISTORY_FILE = BASE_DIR / "chat_history.json"

# Maximum context size (in words) to avoid overloading the model
MAX_CONTEXT_WORDS = 6000

# --- Persistent Storage Functions ---
def load_chat_history():
    """Load chat history from JSON file"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
    return {"sessions": [], "current_session_id": None}

def save_chat_history():
    """Save chat history to JSON file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "sessions": st.session_state.chat_sessions,
                "current_session_id": st.session_state.current_session_id
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    new_session = {
        "id": session_id,
        "name": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    st.session_state.chat_sessions.append(new_session)
    st.session_state.current_session_id = session_id
    save_chat_history()
    return session_id

def get_current_session():
    """Get the current active session"""
    if not st.session_state.current_session_id:
        create_new_session()
    
    for session in st.session_state.chat_sessions:
        if session["id"] == st.session_state.current_session_id:
            return session
    return create_new_session()

def add_message_to_session(role: str, content: str, metadata: Dict = None):
    """Add a message to the current session"""
    session = get_current_session()
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    session["messages"].append(message)
    save_chat_history()
    return message

def rename_current_session(name: str):
    """Rename the current session"""
    session = get_current_session()
    session["name"] = name
    save_chat_history()

def delete_session(session_id: str):
    """Delete a session"""
    st.session_state.chat_sessions = [s for s in st.session_state.chat_sessions if s["id"] != session_id]
    if st.session_state.current_session_id == session_id:
        if st.session_state.chat_sessions:
            st.session_state.current_session_id = st.session_state.chat_sessions[0]["id"]
        else:
            create_new_session()
    save_chat_history()

# --- GPU Configuration ---
def check_gpu_availability():
    """Check if NVIDIA GPU is available and compatible"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            st.sidebar.success(f"🎯 GPU: {gpu.name} ({gpu.memoryFree}MB free)")
            return True
        else:
            st.sidebar.warning("⚠️ No GPU detected - using CPU (slow)")
            return False
    except:
        st.sidebar.error("❌ GPU check failed")
        return False

def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                "name": gpu.name,
                "total_memory": gpu.memoryTotal,  # Keep as float
                "free_memory": gpu.memoryFree,    # Keep as float
                "used_memory": gpu.memoryUsed,    # Keep as float
                "utilization": f"{gpu.load * 100:.1f}%"
            }
        return None
    except:
        return None

def setup_ollama_for_gpu():
    """Configure Ollama to use GPU with optimal settings based on available VRAM"""
    gpu_info = get_gpu_memory_info()
    
    # Determine optimal GPU layers based on available VRAM
    if gpu_info:
        total_vram = gpu_info['total_memory']  # Already a float
        if total_vram >= 24000:  # 24GB+ VRAM (RTX 3090/4090)
            num_gpu_layers = 50
            low_vram = False
        elif total_vram >= 12000:  # 12GB VRAM (RTX 3060-3080)
            num_gpu_layers = 35
            low_vram = False
        elif total_vram >= 8000:   # 8GB VRAM (RTX 4060-4070)
            num_gpu_layers = 25
            low_vram = True
        else:  # Less than 8GB VRAM
            num_gpu_layers = 15
            low_vram = True
    else:
        # Default fallback
        num_gpu_layers = 25
        low_vram = True
    
    gpu_payload = {
        "model": "codellama:7b-instruct-q4_K_M",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 16384,
            "num_predict": 2048,
            "num_gpu": num_gpu_layers,
            "main_gpu": 0,
            "low_vram": low_vram,
        }
    }
    return gpu_payload, num_gpu_layers, low_vram

# --- Ollama Setup Check ---
def check_ollama_running():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    """Try to start Ollama with GPU support"""
    try:
        # Start Ollama with GPU environment
        env = os.environ.copy()
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        time.sleep(5)  # Wait for Ollama to start
        return check_ollama_running()
    except:
        return False

def check_models():
    """Check if required models are available"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            return model_names
        return []
    except:
        return []

def pull_model(model_name):
    """Pull a model if not available"""
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model_name})
        return response.status_code == 200
    except:
        return False

# --- Enhanced Text Extraction Functions ---
def extract_text_from_pdf_enhanced(file_path) -> Tuple[str, Dict]:
    """Enhanced PDF extraction with multiple fallback methods and quality metrics"""
    text = ""
    metadata = {
        "pages_processed": 0,
        "total_pages": 0,
        "extraction_method": "",
        "quality_score": 0,
        "has_tables": False
    }
    
    # Method 1: pdfplumber with table extraction
    try:
        with pdfplumber.open(file_path) as pdf:
            metadata["total_pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text() or ""
                
                # Extract tables if any
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    metadata["has_tables"] = True
                    for table in tables:
                        for row in table:
                            table_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                        table_text += "\n"
                
                # Combine text and tables
                if page_text or table_text:
                    text += f"--- Page {i+1} ---\n{page_text}\n"
                    if table_text:
                        text += f"Tables:\n{table_text}\n"
                    metadata["pages_processed"] += 1
            
            metadata["extraction_method"] = "pdfplumber"
            if text.strip():
                metadata["quality_score"] = min(100, len(text.strip().split()) // 10)  # Rough quality estimate
                return text.strip(), metadata
    except Exception as e:
        st.warning(f"pdfplumber failed for {file_path.name}: {str(e)}")
    
    # Method 2: PyPDF2 fallback
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata["total_pages"] = len(pdf_reader.pages)
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"--- Page {i+1} ---\n{page_text}\n"
                    metadata["pages_processed"] += 1
            
            metadata["extraction_method"] = "PyPDF2"
            if text.strip():
                metadata["quality_score"] = min(100, len(text.strip().split()) // 8)
                return text.strip(), metadata
    except Exception as e:
        st.warning(f"PyPDF2 failed for {file_path.name}: {str(e)}")
    
    return "", metadata

def extract_text_from_docx(file_path) -> str:
    """Extract text from DOCX files"""
    try:
        doc = docx.Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Failed to extract from DOCX {file_path.name}: {str(e)}")
        return ""

def extract_text_from_csv(file_path) -> str:
    """Extract structured data from CSV files"""
    try:
        df = pd.read_csv(file_path)
        # Convert to readable text format
        text = f"CSV Data from {file_path.name}:\n"
        text += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        text += "First 10 rows:\n"
        text += df.head(10).to_string()
        return text
    except Exception as e:
        st.error(f"Failed to read CSV {file_path.name}: {str(e)}")
        return ""

def extract_text_from_image(file_path) -> str:
    """Extract text from images using OCR"""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"OCR failed for {file_path.name}: {str(e)}")
        return ""

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Fix common OCR/PDF issues
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
    text = re.sub(r'\uf0b7', '•', text)  # Fix bullet points
    
    return text.strip()

def smart_text_truncation(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str:
    """Intelligently truncate text while preserving important sections"""
    if len(text.split()) <= max_words:
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Keep beginning and end, remove middle
    keep_each_side = max_words // 2
    truncated = words[:keep_each_side] + ["\n\n...[Content truncated for length...]\n\n"] + words[-keep_each_side:]
    return " ".join(truncated)

# --- Enhanced File Processing ---
def process_single_file(file_path):
    """Process a single file and return extracted text and metadata"""
    extracted_text = ""
    metadata = {}
    
    try:
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            extracted_text, metadata = extract_text_from_pdf_enhanced(file_path)
        elif file_ext in ['.docx']:
            extracted_text = extract_text_from_docx(file_path)
            metadata = {"extraction_method": "python-docx", "quality_score": 95}
        elif file_ext == '.csv':
            extracted_text = extract_text_from_csv(file_path)
            metadata = {"extraction_method": "pandas", "quality_score": 90}
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            extracted_text = extract_text_from_image(file_path)
            metadata = {"extraction_method": "OCR", "quality_score": 70}
        else:  # Text files
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
            metadata = {"extraction_method": "direct", "quality_score": 100}
        
        # Clean and validate extracted text
        if extracted_text:
            extracted_text = clean_extracted_text(extracted_text)
            if len(extracted_text.strip()) > 10:  # Minimum viable text
                return {
                    "name": file_path.name,
                    "path": file_path,
                    "text": extracted_text,
                    "metadata": metadata,
                    "word_count": len(extracted_text.split())
                }
    
    except Exception as e:
        st.warning(f"Error processing {file_path.name}: {str(e)}")
    
    return None

def process_existing_files():
    """Process all existing files in upload directory on app start"""
    current_files = list(UPLOAD_DIR.glob("*"))
    if not current_files:
        return
    
    # Check if we need to process any files
    unprocessed_files = [f for f in current_files if f.name not in st.session_state.processed_files]
    if not unprocessed_files:
        return
    
    with st.sidebar:
        with st.status("🔄 Processing existing documents...", expanded=True) as status:
            for i, file_path in enumerate(unprocessed_files):
                status.write(f"Processing {file_path.name}...")
                result = process_single_file(file_path)
                if result:
                    st.session_state.processed_files[result['name']] = result
                    st.session_state.file_metadata.append(result['metadata'])
                time.sleep(0.5)  # Small delay to show progress
            
            status.update(label="✅ All documents processed!", state="complete", expanded=False)

def process_uploaded_files(uploaded_files) -> Dict:
    """Process multiple files with progress tracking and quality assessment"""
    results = {
        "successful": [],
        "failed": [],
        "total_files": len(uploaded_files),
        "total_text_length": 0
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        progress_bar.progress((i) / len(uploaded_files))
        
        file_path = save_uploaded_file(uploaded_file)
        result = process_single_file(file_path)
        
        if result:
            results["successful"].append(result)
            results["total_text_length"] += result['word_count']
        else:
            results["failed"].append({
                "name": uploaded_file.name,
                "reason": "No text could be extracted"
            })
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return results

def read_all_documents() -> Tuple[str, List[Dict]]:
    """Read all documents with enhanced processing"""
    text_content = []
    file_metadata = []
    
    for file_name, file_data in st.session_state.processed_files.items():
        text_content.append(f"--- {file_data['name']} ---\n{file_data['text']}")
        file_metadata.append({
            "name": file_data['name'],
            "type": Path(file_data['name']).suffix.lower(),
            "word_count": file_data['word_count'],
            "metadata": file_data['metadata']
        })
    
    combined_text = "\n\n".join(text_content)
    # Apply smart truncation if needed
    combined_text = smart_text_truncation(combined_text)
    
    return combined_text, file_metadata

# --- Enhanced RAG with Ollama ---
def query_ollama(prompt: str, context: str = "", document_metadata: List[Dict] = None) -> str:
    """GPU-optimized RAG query with better prompting"""
    
    # Build document info string
    doc_info = ""
    if document_metadata:
        doc_info = "Documents analyzed:\n" + "\n".join([
            f"- {doc['name']} ({doc['type']}, {doc['word_count']} words)" 
            for doc in document_metadata
        ]) + "\n\n"
    
    full_prompt = f"""Based on the following documents, please provide a comprehensive and accurate answer to the question.

{doc_info}DOCUMENT CONTENT:
{context}

QUESTION: {prompt}

Please:
1. Base your answer strictly on the provided documents
2. If the documents don't contain relevant information, clearly state this
3. Provide specific references to document content when possible
4. Be thorough but concise

ANSWER:"""
    
    # Use GPU-optimized payload
    payload, num_gpu_layers, low_vram = setup_ollama_for_gpu()
    payload["prompt"] = full_prompt
    
    # Store GPU settings in session for display
    st.session_state.gpu_settings = {
        'num_gpu_layers': num_gpu_layers,
        'low_vram': low_vram
    }
    
    try:
        with st.spinner(f"🔄 Generating answer using GPU ({num_gpu_layers} layers)..."):
            response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
        if response.status_code == 200:
            return response.json().get('response', 'No response received')
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# --- File Management ---
def save_uploaded_file(uploaded_file):
    """Save uploaded file with unique naming"""
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def clear_uploaded_files():
    """Clear all uploaded files"""
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            file_path.unlink()

def get_file_stats():
    """Get detailed statistics about uploaded files"""
    files = list(UPLOAD_DIR.glob("*"))
    stats = {
        'total': len(files),
        'pdfs': len([f for f in files if f.suffix.lower() == '.pdf']),
        'docx': len([f for f in files if f.suffix.lower() == '.docx']),
        'images': len([f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']]),
        'csv': len([f for f in files if f.suffix.lower() == '.csv']),
        'text_files': len([f for f in files if f.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']])
    }
    return stats

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="🎯 GPU-Optimized Document Q&A", layout="wide")
    st.title("🎯 GPU-Optimized Multi-Document Q&A")
    
    # Initialize session state with persistent chat history
    if 'chat_sessions' not in st.session_state:
        history_data = load_chat_history()
        st.session_state.chat_sessions = history_data.get("sessions", [])
        st.session_state.current_session_id = history_data.get("current_session_id")
    
    # Initialize other session state variables
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'file_metadata' not in st.session_state:
        st.session_state.file_metadata = []
    if 'gpu_settings' not in st.session_state:
        st.session_state.gpu_settings = {}
    
    # Create initial session if none exists
    if not st.session_state.chat_sessions:
        create_new_session()
    
    # Process existing files on app start
    process_existing_files()
    
    # === SIDEBAR CONTENT ===
    
    # GPU Status Check
    st.sidebar.header("🎯 GPU Status")
    gpu_available = check_gpu_availability()
    gpu_info = get_gpu_memory_info()
    
    if gpu_info:
        with st.sidebar.expander("📊 GPU Details", expanded=True):
            st.write(f"**Name:** {gpu_info['name']}")
            st.write(f"**Total VRAM:** {gpu_info['total_memory']:.0f}MB")
            st.write(f"**Free VRAM:** {gpu_info['free_memory']:.0f}MB")
            st.write(f"**Utilization:** {gpu_info['utilization']}")
            
            # Show current GPU settings
            if st.session_state.gpu_settings:
                st.write(f"**GPU Layers:** {st.session_state.gpu_settings.get('num_gpu_layers', 'N/A')}")
                st.write(f"**Low VRAM Mode:** {st.session_state.gpu_settings.get('low_vram', 'N/A')}")
    
    # Ollama status check
    st.sidebar.header("🔧 Ollama Status")
    
    if not check_ollama_running():
        st.sidebar.error("❌ Ollama not running")
        if st.sidebar.button("Try to Start Ollama Automatically"):
            if start_ollama():
                st.sidebar.success("✅ Ollama started!")
                st.rerun()
            else:
                st.sidebar.error("Failed to start Ollama automatically")
        return
    
    st.sidebar.success("✅ Ollama is running!")
    
    # Model check
    available_models = check_models()
    required_model = "codellama:7b-instruct-q4_K_M"
    
    if required_model not in available_models:
        st.sidebar.warning(f"❌ Model '{required_model}' not found")
        if st.sidebar.button(f"Download {required_model}"):
            with st.sidebar.status("Downloading model...", expanded=True) as status:
                st.write("🔄 Downloading model with GPU support...")
                if pull_model(required_model):
                    status.update(label="Download complete!", state="complete")
                    st.rerun()
                else:
                    status.update(label="Download failed", state="error")
    else:
        st.sidebar.success(f"✅ Model '{required_model}' available")
    
    # Chat Sessions Management
    st.sidebar.header("💬 Chat Sessions")
    
    with st.sidebar.expander("Session Management", expanded=True):
        # Create new session button
        if st.button("➕ New Chat", use_container_width=True):
            create_new_session()
            st.rerun()
        
        # Session list
        current_session = get_current_session()
        for session in st.session_state.chat_sessions:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                is_current = session["id"] == st.session_state.current_session_id
                if st.button(
                    f"💬 {session['name']}", 
                    key=f"session_{session['id']}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    st.session_state.current_session_id = session["id"]
                    st.rerun()
            
            with col2:
                if len(st.session_state.chat_sessions) > 1 and not is_current:
                    if st.button("🗑️", key=f"delete_{session['id']}", help="Delete session"):
                        delete_session(session["id"])
                        st.rerun()
    
    # Document Library in Sidebar
    current_files = list(UPLOAD_DIR.glob("*"))
    if current_files:
        st.sidebar.header("📋 Document Library")
        
        with st.sidebar.expander("View Documents", expanded=True):
            for file in current_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    file_icon = "📄" if file.suffix.lower() == '.pdf' else "📝"
                    st.write(f"{file_icon} {file.name}")
                with col2:
                    if file.name in st.session_state.processed_files:
                        st.success("✅")
                    else:
                        st.warning("⏳")
    
    # Document statistics
    if current_files:
        stats = get_file_stats()
        total_words = sum(item['word_count'] for item in st.session_state.processed_files.values())
        
        st.sidebar.header("📊 Document Stats")
        st.sidebar.metric("Total Files", stats['total'])
        st.sidebar.metric("Total Words", f"{total_words:,}")
        st.sidebar.metric("PDFs", stats['pdfs'])
        st.sidebar.metric("Text Files", stats['text_files'])
        
        if stats['docx'] > 0:
            st.sidebar.metric("Word Docs", stats['docx'])
        if stats['images'] > 0:
            st.sidebar.metric("Images", stats['images'])
        if stats['csv'] > 0:
            st.sidebar.metric("CSV Files", stats['csv'])
    
    # Management section in Sidebar
    st.sidebar.header("🛠️ Management")
    with st.sidebar.expander("Document Controls", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Docs", use_container_width=True, help="Clear all documents"):
                clear_uploaded_files()
                st.session_state.processed_files = {}
                st.session_state.file_metadata = []
                st.success("✅ All documents cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Reprocess", use_container_width=True, help="Reprocess all documents"):
                st.session_state.processed_files = {}
                st.session_state.file_metadata = []
                process_existing_files()
                st.rerun()
        
        if st.button("📊 System Info", use_container_width=True):
            gpu_info = get_gpu_memory_info()
            system_info = f"""
            **System Status:**
            - Ollama: ✅ Running
            - Model: {required_model}
            - Max Context: {MAX_CONTEXT_WORDS} words
            - Upload Dir: {UPLOAD_DIR}
            - Chat Sessions: {len(st.session_state.chat_sessions)}
            """
            if gpu_info:
                system_info += f"""
                **GPU Performance:**
                - GPU: {gpu_info['name']}
                - VRAM: {gpu_info['total_memory']:.0f}MB total, {gpu_info['free_memory']:.0f}MB free
                - Utilization: {gpu_info['utilization']}
                - GPU Layers: {st.session_state.gpu_settings.get('num_gpu_layers', 'N/A')}
                """
            st.sidebar.info(system_info)

    # Enhanced tips section
    with st.sidebar.expander("💡 Tips & Features"):
        st.write("""
        **Persistent Features:**
        - Chat history saved automatically
        - Multiple chat sessions supported
        - Session management (create/delete/switch)
        - All data persists across page refreshes
        
        **GPU Performance:**
        - Model runs primarily on GPU
        - Automatic VRAM optimization
        - Fast document processing
        
        **Supported Formats:**
        - 📄 PDF, 📝 DOCX/TXT
        - 📊 CSV, 🖼️ Images (OCR)
        - 💻 Code files
        """)
    
    # === MAIN CONTENT ===
    
    # Current session info
    current_session = get_current_session()
    st.header(f"💬 {current_session['name']}")
    
    # Rename session
    with st.expander("✏️ Rename Session"):
        new_name = st.text_input("Session name:", value=current_session['name'])
        if st.button("Update Name") and new_name != current_session['name']:
            rename_current_session(new_name)
            st.success("Session name updated!")
            st.rerun()
    
    # File upload section
    st.header("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to analyze (PDF, DOCX, TXT, CSV, Images, etc.)",
        type=['pdf', 'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'csv', 'docx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple files of different types for comprehensive analysis"
    )
    
    if uploaded_files:
        if st.button("🚀 Process All Files", type="primary"):
            with st.spinner("Processing files with enhanced extraction..."):
                results = process_uploaded_files(uploaded_files)
                
                # Display results
                st.success(f"✅ Successfully processed {len(results['successful'])}/{len(uploaded_files)} files")
                
                if results['failed']:
                    with st.expander("❌ Failed Files", expanded=False):
                        for failed in results['failed']:
                            st.error(f"{failed['name']}: {failed['reason']}")
                
                # Store processed files
                for item in results["successful"]:
                    st.session_state.processed_files[item['name']] = item
                    st.session_state.file_metadata.append(item['metadata'])
    
    # Display chat history for current session
    st.header("📝 Conversation")
    
    # Display messages from current session
    for message in current_session["messages"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
                if message["metadata"].get("documents_used"):
                    st.caption(f"📚 Used {message['metadata']['documents_used']} documents")
        else:  # assistant
            with st.chat_message("assistant"):
                st.write(message["content"])
                if message["metadata"]:
                    with st.expander("🔍 Response Details"):
                        st.write(f"**Context Size:** {message['metadata'].get('context_size', 'N/A')} words")
                        st.write(f"**GPU Layers:** {message['metadata'].get('gpu_layers', 'N/A')}")
                        st.write(f"**Timestamp:** {datetime.fromisoformat(message['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Query section
    st.header("❓ Ask a Question")
    
    # Enhanced quick action buttons (only show if documents are loaded)
    if current_files:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Summarize All", help="Comprehensive summary across all documents", use_container_width=True):
                st.session_state.auto_query = "Provide a detailed summary of all documents, highlighting main themes, key findings, and connections between documents"
        with col2:
            if st.button("🔍 Compare Documents", help="Compare and contrast multiple documents", use_container_width=True):
                st.session_state.auto_query = "Compare and contrast the main points across all documents. What are the similarities, differences, and relationships?"
        with col3:
            if st.button("📊 Extract Key Insights", help="Extract most important insights", use_container_width=True):
                st.session_state.auto_query = "What are the most important insights, findings, and recommendations across all documents?"
    
    # Chat input
    query = st.chat_input("Ask a question about your documents...", key="chat_input")
    
    # Use auto-query if set
    if hasattr(st.session_state, 'auto_query'):
        query = st.session_state.auto_query
        del st.session_state.auto_query
    
    if query:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(query)
        
        if not current_files:
            with st.chat_message("assistant"):
                st.error("❌ Please upload and process some documents first")
        else:
            with st.spinner("🔍 Analyzing documents with GPU-accelerated RAG..."):
                # Read all documents with enhanced processing
                context, doc_metadata = read_all_documents()
                
                if not context.strip():
                    with st.chat_message("assistant"):
                        st.error("❌ No readable text found in uploaded documents")
                else:
                    # Show enhanced context stats
                    word_count = len(context.split())
                    char_count = len(context)
                    doc_count = len(doc_metadata)
                    
                    # Get analysis from Ollama
                    answer = query_ollama(query, context, doc_metadata)
                    
                    # Add messages to persistent storage
                    add_message_to_session("user", query, {
                        "documents_used": doc_count,
                        "context_size": word_count
                    })
                    
                    add_message_to_session("assistant", answer, {
                        "documents_used": doc_count,
                        "context_size": word_count,
                        "gpu_layers": st.session_state.gpu_settings.get('num_gpu_layers', 'N/A')
                    })
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(answer)
                        with st.expander("🔍 Response Details"):
                            st.write(f"**Documents Analyzed:** {doc_count}")
                            st.write(f"**Context Size:** {word_count} words")
                            st.write(f"**GPU Layers:** {st.session_state.gpu_settings.get('num_gpu_layers', 'N/A')}")
            
            # Rerun to update the display
            st.rerun()

if __name__ == "__main__":
    main()