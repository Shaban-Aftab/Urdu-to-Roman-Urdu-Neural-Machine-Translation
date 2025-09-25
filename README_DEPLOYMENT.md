# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your Urdu to Roman Transliteration app to Streamlit Cloud using your GitHub repository.

## 📋 Prerequisites

1. **GitHub Account**: Ensure you have access to your repository: `https://github.com/Shaban-Aftab/Urdu-to-Roman-Urdu-Neural-Machine-Translation.git`
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Deployment Files Prepared

The following files have been created/updated for deployment:

### 1. **app_deployment.py** 
- Deployment-ready version of your app
- Uses rule-based transliteration as fallback (no model files needed)
- Optimized for cloud deployment

### 2. **requirements.txt** (Updated)
- Minimal dependencies for faster deployment
- CPU-only PyTorch version
- Streamlit and essential packages only

### 3. **.streamlit/config.toml**
- Streamlit configuration for deployment
- Custom theme matching your app design
- Optimized server settings

## 📤 Step-by-Step Deployment Process

### Step 1: Push Files to GitHub

1. **Add the new files to your repository:**
   ```bash
   git add app_deployment.py
   git add requirements.txt
   git add .streamlit/config.toml
   git add README_DEPLOYMENT.md
   ```

2. **Commit the changes:**
   ```bash
   git commit -m "Add deployment files for Streamlit Cloud"
   ```

3. **Push to GitHub:**
   ```bash
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App:**
   - Click "New app"
   - Select "From existing repo"

3. **Configure Deployment:**
   - **Repository:** `Shaban-Aftab/Urdu-to-Roman-Urdu-Neural-Machine-Translation`
   - **Branch:** `main` (or your default branch)
   - **Main file path:** `app_deployment.py`
   - **App URL:** Choose a custom URL (optional)

4. **Deploy:**
   - Click "Deploy!"
   - Wait for the deployment to complete (usually 2-5 minutes)

### Step 3: Verify Deployment

1. **Check App Status:**
   - Your app should be accessible at the provided URL
   - Test the transliteration functionality
   - Verify all UI elements are working

2. **Monitor Logs:**
   - Use the Streamlit Cloud dashboard to monitor logs
   - Check for any deployment issues

## 🎯 App Features (Deployment Version)

### ✅ **Working Features:**
- **Rule-based Transliteration**: Immediate character mapping
- **Beautiful UI**: Custom CSS styling and responsive design
- **Interactive Examples**: Quick-start buttons with sample text
- **Text Statistics**: Real-time character, word, and sentence counts
- **Download Functionality**: Save results as text files
- **Mobile Responsive**: Works on all device sizes

### 📝 **Demo Notice:**
The deployment version uses rule-based transliteration instead of the neural model because:
- Model files (`best_model_baseline.pth`, tokenizer files) are too large for GitHub
- Streamlit Cloud has resource limitations
- Rule-based approach provides instant results

## 🔄 Alternative Deployment Options

### Option 1: With Model Files (Advanced)

If you want to deploy the full neural model:

1. **Use Git LFS for large files:**
   ```bash
   git lfs track "*.pth"
   git lfs track "*.model"
   git add .gitattributes
   ```

2. **Update app.py to use the original version**

3. **Increase resource requirements in Streamlit Cloud**

### Option 2: Hugging Face Spaces

Consider deploying to Hugging Face Spaces for better support of large model files:

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space with Streamlit
3. Upload your files including model weights
4. Deploy with higher resource allocation

## 🛠️ Troubleshooting

### Common Issues:

1. **Deployment Fails:**
   - Check requirements.txt for version conflicts
   - Ensure all files are properly committed to GitHub
   - Verify the main file path is correct

2. **App Crashes:**
   - Check Streamlit Cloud logs
   - Verify all imports are available
   - Test locally first with `streamlit run app_deployment.py`

3. **Slow Loading:**
   - Streamlit Cloud may take time for cold starts
   - Consider upgrading to Streamlit Cloud Pro for better performance

### Local Testing:

Before deploying, test locally:
```bash
pip install -r requirements.txt
streamlit run app_deployment.py
```

## 📞 Support

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Use your repository's issues section

## 🎉 Success!

Once deployed, your app will be accessible worldwide at your Streamlit Cloud URL. Share the link with others to showcase your Urdu to Roman transliteration project!

---

**Note**: This deployment uses a demo version with rule-based transliteration. For production use with the full neural model, consider using cloud platforms with higher resource limits or implement model serving solutions like FastAPI + Docker.