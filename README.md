# Resident Scheduler

This repository contains the editable Streamlit prototype that powers the resident scheduling tool.

## Local development

1. Create a virtual environment (optional but recommended) and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Streamlit:
   ```bash
   streamlit run app.py --server.port 8501 --server.address localhost
   ```

The `--server.address localhost` flag keeps Streamlit bound to the container/VM; the IDE's port-forwarding
feature exposes the service externally.

## Viewing the UI

`http://0.0.0.0:8501` is only reachable **inside** the dev container. When you are using a cloud IDE, open
port 8501 via the platform's "Open in Browser" / "Webview" button so it forwards a public URL (usually something
like `https://<workspace>-8501.proxy...`). If you try to visit `http://0.0.0.0:8501` directly from your local
browser you'll see "site can't be reached" because that address is not routable on your machine.

## Password

The app checks `st.secrets["APP_PASSWORD"]` first and then `APP_PASSWORD` from the environment. Leaving it unset
enables local development: Streamlit will show a warning but immediately unlock the UI.
