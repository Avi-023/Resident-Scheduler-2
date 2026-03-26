# Surgery Resident Scheduler

A Streamlit web application for generating and managing surgical resident rotation schedules.

## Features

- **Roster Management**: Edit resident information including PGY level, vacation dates
- **Automatic Schedule Generation**: Greedy algorithm assigns rotations fairly
- **Daily Block View**: View and edit daily assignments with dropdown menus
- **Weekend Call Assignment**: Automatic fair distribution of F/Su and Sa call
- **Vacation Auto-Fill**: Specify vacation weeks in roster, auto-fills daily blocks
- **Compliance Checks**: Verify 4+ days off per block, vacation weeks, personal days
- **Calendar Subscriptions**: Residents subscribe to auto-updating calendars
- **ACGME Case Logs Dashboard**: Track resident progress toward graduation requirements
- **Email Import**: Coordinator emails case logs → automatically imported via Google Drive
- **Excel/CSV Export**: Export schedules in various formats

## Deployment to Streamlit Cloud

### Prerequisites

1. A GitHub account
2. This repository pushed to GitHub
3. A [Streamlit Cloud](https://streamlit.io/cloud) account (free tier available)

### Step-by-Step Deployment

1. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Resident-Scheduler-2.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set the main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to your app settings
   - Click "Secrets" in the left sidebar
   - Add your secrets in TOML format:
     ```toml
     APP_PASSWORD = "your-secure-password-here"
     ```
   - Click "Save"

4. **Access Your App**
   - Your app will be available at: `https://your-app-name.streamlit.app`
   - Share this URL with your residents

### Updating the App

Simply push changes to your GitHub repository:
```bash
git add .
git commit -m "Update description"
git push
```
Streamlit Cloud will automatically redeploy.

## Local Development

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Resident-Scheduler-2.git
   cd Resident-Scheduler-2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # Edit .streamlit/secrets.toml with your password
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   The app will open at http://localhost:8501

## Configuration

### Streamlit Config (`.streamlit/config.toml`)

```toml
[server]
headless = true
runOnSave = false

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#4F46E5"
```

### Secrets (`.streamlit/secrets.toml`)

**Never commit this file!** Configure in Streamlit Cloud dashboard or locally.

```toml
APP_PASSWORD = "your-password"
```

## Email Import Setup (Google Drive)

Enable automatic case log import via email. When the coordinator emails an ACGME export, it's automatically imported into the dashboard.

### How It Works

```
Coordinator exports ACGME case logs (PDF)
            ↓
Emails to designated inbox (e.g., caselogs@gmail.com)
            ↓
Zapier saves attachment to Google Drive folder
            ↓
App auto-imports on next page load
```

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., "Resident Scheduler")
3. Enable the **Google Drive API**:
   - Go to APIs & Services → Library
   - Search "Google Drive API" → Enable

### Step 2: Create Service Account

1. Go to APIs & Services → Credentials
2. Click **Create Credentials** → **Service Account**
3. Name it (e.g., "streamlit-app")
4. Click **Done**
5. Click on the service account → **Keys** tab
6. **Add Key** → **Create new key** → **JSON**
7. Save the downloaded JSON file securely

### Step 3: Set Up Google Drive Folder

1. Create a new folder in Google Drive (e.g., "ACGME Case Logs")
2. Right-click → **Share**
3. Add the service account email (from the JSON file, looks like: `name@project.iam.gserviceaccount.com`)
4. Give it **Viewer** access
5. Copy the folder ID from the URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`

### Step 4: Set Up Zapier (Free Tier)

1. Go to [Zapier](https://zapier.com) and create account
2. Create a new Zap:
   - **Trigger**: Gmail → New Attachment
   - **Filter**: Only from coordinator's email (optional)
   - **Action**: Google Drive → Upload File
   - Set destination folder to your shared folder

### Step 5: Configure Streamlit Secrets

In Streamlit Cloud dashboard (or local `.streamlit/secrets.toml`):

```toml
APP_PASSWORD = "your-password"

[google_drive]
type = "service_account"
project_id = "your-project-id"
private_key_id = "abc123..."
private_key = "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----\n"
client_email = "streamlit-app@your-project.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
folder_id = "your-google-drive-folder-id"
```

Copy values from the downloaded JSON file. The `folder_id` is the ID from step 3.

### Testing

1. Have the coordinator email a test PDF to the designated inbox
2. Wait for Zapier to process (usually under 1 minute)
3. Go to Case Logs tab and click "Check for new files"
4. The case log data should appear in the dashboard

## Usage Guide

### For Administrators

1. **Initial Setup**
   - Log in with the configured password
   - Enter resident roster in the Yearly tab
   - Set vacation dates in Vacation 1/2/3 columns (e.g., "Block 3", "7/1/2025")

2. **Generate Schedule**
   - Click "Generate schedule from roster"
   - Review the yearly schedule
   - Make manual adjustments if needed
   - Click "Apply Yearly edits" to rebuild daily blocks

3. **Export & Distribute**
   - Go to Export tab
   - Download Excel for record keeping
   - Download calendar ZIP to distribute to residents

### For Residents

1. Receive your calendar file (.ics) from the administrator
2. Import into your calendar app:
   - **Google Calendar**: Settings → Import & Export → Import
   - **Outlook**: File → Open & Export → Import/Export
   - **Apple Calendar**: File → Import

## File Structure

```
Resident-Scheduler-2/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── .streamlit/
    ├── config.toml             # Streamlit configuration
    ├── secrets.toml            # Local secrets (not in git)
    └── secrets.toml.example    # Example secrets template
```

## Troubleshooting

### "No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### Excel export not working
```bash
pip install openpyxl xlsxwriter
```

### Password not working on Streamlit Cloud
- Verify secrets are configured in the Streamlit Cloud dashboard
- Check that `APP_PASSWORD` is set correctly (case-sensitive)

## License

Internal use only - Conemaugh Surgery Department
