# Nirmaan Netra - Setup Guide

## Changes Made to app.py

### 1. ✅ Removed Hardcoded System Path
**Before:**
```python
import sys
sys.path.append('C:\\Users\\ekamb\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages')
```

**After:**
Removed - Python automatically manages the path

**Why:** The hardcoded path was specific to one user and would fail on other machines.

---

### 2. ✅ Secured Secret Key
**Before:**
```python
app.secret_key = '1234567890'  # Weak!
```

**After:**
```python
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
```

**Why:** The secret key is used to sign sessions. A weak key means session hijacking. Now it:
- Reads from `.env` file (production)
- Generates a random 64-character key if not set (development)

---

### 3. ✅ Secured API Key
**Before:**
```python
DEEPAI_API_KEY = 'e6ee2a29-7dc9-4204-bc1c-ffe3c8500bb6'  # Exposed!
```

**After:**
```python
DEEPAI_API_KEY = os.environ.get('DEEPAI_API_KEY', '')
```

**Why:** Never commit secrets to version control. Now it:
- Reads from `.env` file
- Falls back to empty string if not configured

---

### 4. ✅ Added Environment Variable Loading
Added `python-dotenv` support to load `.env` file:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Files Created

### `.env` (Local Configuration)
Contains your local settings. **Never commit this to Git!** Add `.env` to `.gitignore`.

```
SECRET_KEY=dev-secret-key-change-in-production
DEEPAI_API_KEY=your_api_key_here
```

### `.env.example` (Documentation)
Shows what environment variables are needed. Safe to commit to Git.

---

## Setup Instructions

### 1. Install python-dotenv
```bash
pip install python-dotenv
```

### 2. Configure your `.env` file
Edit `.env` and set:
- `SECRET_KEY` - Leave as is for development
- `DEEPAI_API_KEY` - [Get from deepai.org](https://deepai.org/)

### 3. Add `.env` to `.gitignore`
```bash
echo ".env" >> .gitignore
```

---

## ML Models
The three models have been created:
- ✅ `models/RIO_building_detect_model1.keras`
- ✅ `models/water_model_heavy_rgb.h5`
- ✅ `models/unet_building_change_detection1.h5`

These are **untrained placeholder models** for testing. Replace with trained models for production.

---

## Running the App

```bash
python app.py
```

The app should now:
- ✅ Load without hardcoded paths
- ✅ Have secure secret keys
- ✅ Properly manage API keys
- ✅ Find the ML models in the `models/` directory

---

## Security Checklist

- [ ] Install `python-dotenv`: `pip install python-dotenv`
- [ ] Set up `.env` with your API keys
- [ ] Add `.env` to `.gitignore`
- [ ] Don't commit secrets to Git
- [ ] Use strong `SECRET_KEY` in production
- [ ] Replace placeholder ML models with trained versions

