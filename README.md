# Surgery Resident Scheduler

A Streamlit web application for generating and managing surgical resident rotation schedules.

## Features

- **Roster Management**: Edit resident information including PGY level, vacation dates
- **Automatic Schedule Generation**: Greedy algorithm assigns rotations fairly
- **Daily Block View**: View and edit daily assignments with dropdown menus
- **Weekend Call Assignment**: Automatic fair distribution of F/Su and Sa call
- **Vacation Auto-Fill**: Specify vacation weeks in roster, auto-fills daily blocks
- **Compliance Checks**: Verify 4+ days off per block, vacation weeks, personal days
- **Calendar Export**: Download .ics files for Google Calendar, Outlook, Apple Calendar
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
