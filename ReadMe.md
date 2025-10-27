# 🚀 Persistent Multi-Document RAG System

A powerful, GPU-optimized Retrieval-Augmented Generation (RAG) system that enables intelligent document analysis with persistent chat history and multi-format support.

## 📖 Overview

This advanced RAG system allows users to upload multiple document formats (PDF, DOCX, CSV, images, etc.) and ask questions about their content using AI-powered analysis. The system features persistent chat sessions, GPU acceleration, and a ChatGPT-like interface.

## ✨ Features

### 🎯 Core Capabilities
- **Multi-Format Document Processing**: Support for PDF, DOCX, TXT, CSV, images (OCR), and code files
- **Persistent Chat Sessions**: ChatGPT-like interface with history that survives page refreshes
- **GPU Acceleration**: Automatic optimization for NVIDIA RTX cards with dynamic VRAM management
- **Smart Context Management**: Advanced text extraction and intelligent context truncation

### 💬 Chat Experience
- **Multiple Session Support**: Create, switch between, and manage multiple chat sessions
- **Persistent History**: All conversations saved automatically and restored on app restart
- **Session Management**: Rename, delete, and organize chat sessions
- **Real-time Updates**: Live chat interface with streaming-like response display

### 🔧 Technical Features
- **Advanced Text Extraction**: Multi-method PDF parsing with table recognition
- **OCR Support**: Text extraction from images using Tesseract
- **Auto-Processing**: Documents automatically processed on app startup
- **Performance Monitoring**: Real-time GPU and system statistics

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) with CUDA support
- Ollama installed and running

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/persistent-rag-system.git
cd persistent-rag-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv # On Windows: venv\Scripts\activate
```

```bash
source venv/Scripts/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install GPU Dependencies (Optional but Recommended)
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Ollama
Follow instructions from [Ollama Official Website](https://ollama.ai) for your operating system.

### 6. Pull Required Model
```bash
ollama pull codellama:7b-instruct-q4_K_M
```

## 🚀 Usage

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Launch the Application
```bash
streamlit run main.py
```

### 3. Access the Application
Open your browser and navigate to `http://localhost:8501`

### 4. Using the System
1. **Upload Documents**: Use the file uploader to add PDFs, DOCX, CSV, or image files
2. **Process Files**: Click "Process All Files" to extract text content
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **Manage Sessions**: Create new chats or switch between existing sessions in the sidebar

## 📁 Project Structure

```
persistent-rag-system/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── chat_history.json       # Persistent chat storage (auto-generated)
├── uploaded_docs/          # Document storage directory (auto-generated)
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## 🔧 Configuration

### GPU Optimization
The system automatically detects your GPU and optimizes settings based on available VRAM:
- **24GB+ VRAM**: 50 GPU layers (RTX 3090/4090)
- **12GB VRAM**: 35 GPU layers (RTX 3060-3080)
- **8GB VRAM**: 25 GPU layers with low VRAM mode (RTX 4060-4070)
- **<8GB VRAM**: 15 GPU layers with low VRAM mode

### Supported File Formats
- **PDF**: Text and table extraction with fallback methods
- **DOCX**: Structured document parsing
- **CSV**: Data preview and structured analysis
- **Images**: OCR text extraction (PNG, JPG, JPEG, TIFF, BMP)
- **Text Files**: TXT, MD, Python, JS, HTML, CSS, JSON

## 🎯 Key Components

### Persistent Storage
- **Chat History**: JSON-based storage of all conversations and sessions
- **Document Cache**: Processed text storage for fast access
- **Session Management**: UUID-based session tracking

### AI Integration
- **Model**: CodeLlama 7B Instruct (q4_K_M quantization)
- **API**: Ollama local inference server
- **Prompt Engineering**: Document-aware context optimization

### Text Processing
- **PDF Extraction**: pdfplumber (primary) + PyPDF2 (fallback)
- **OCR**: Tesseract for image text recognition
- **Text Cleaning**: Advanced normalization and formatting
- **Smart Truncation**: Context-aware text shortening

## 🔄 API Endpoints

The system uses Ollama's local API:
- **Base URL**: `http://localhost:11434`
- **Generate Endpoint**: `/api/generate`
- **Model Management**: `/api/tags`, `/api/pull`

## 🚀 Deployment

### Local Development
```bash
# Start Ollama in one terminal
ollama serve

# Start Streamlit app in another terminal
streamlit run main.py
```

### Production Considerations
- Ensure Ollama service is always running
- Configure GPU drivers and CUDA properly
- Set up proper file permissions for upload directories
- Consider using process managers (systemd, pm2) for long-running deployment

## 📊 Performance

### Expected Performance by GPU
- **RTX 4090/3090**: 2-5 second response times
- **RTX 3080**: 3-7 second response times  
- **RTX 4060/4070**: 5-10 second response times
- **CPU Only**: 15-30 second response times

### Memory Usage
- **Model**: ~4.1GB (codellama:7b-instruct-q4_K_M)
- **Documents**: Varies by file size and quantity
- **Chat History**: Minimal storage requirements

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Not Running**
   ```bash
   # Check if Ollama is running
   ollama ps
   # Start if not running
   ollama serve
   ```

2. **GPU Not Detected**
   - Verify NVIDIA drivers are installed
   - Check CUDA installation
   - System will fall back to CPU mode

3. **Model Not Found**
   ```bash
   # Pull the required model
   ollama pull codellama:7b-instruct-q4_K_M
   ```

4. **File Upload Issues**
   - Check directory permissions for `uploaded_docs/`
   - Verify file formats are supported
   - Ensure files are not corrupted

### Logs and Debugging
- Check Streamlit logs in the terminal
- Verify Ollama logs for model inference issues
- Monitor GPU utilization in the sidebar stats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for providing the local inference engine
- [Streamlit](https://streamlit.io) for the web framework
- [CodeLlama](https://ai.meta.com/blog/code-llama-large-language-model-coding/) for the AI model
- All the open-source libraries that made this project possible

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**⭐ If you find this project useful, please give it a star on GitHub!**
