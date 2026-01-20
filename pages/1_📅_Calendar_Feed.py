# pages/1_📅_Calendar_Feed.py
"""
Calendar subscription page - serves ICS calendar feeds for residents.

Residents can subscribe to their calendar using a URL like:
https://your-app.streamlit.app/Calendar_Feed?resident=Smith_John

Calendar apps will periodically fetch this URL to get updates.
"""

import streamlit as st
import re
import time
from datetime import date, timedelta

st.set_page_config(page_title="Calendar Feed", page_icon="📅", layout="centered")

# Get query parameters
params = st.query_params

# Check if this is a calendar subscription request
resident_param = params.get("resident", None)
download_mode = params.get("download", None)

def get_resident_list():
    """Get list of residents from stored dailies."""
    dailies = st.session_state.get("dailies", {})
    if not dailies:
        return []
    residents = set()
    for df in dailies.values():
        residents.update(df.index)
    return sorted([r for r in residents if isinstance(r, str) and r.strip() and r.lower() != "none"])

def normalize_resident_name(name: str) -> str:
    """Convert URL-safe name back to display name."""
    return name.replace("_", " ").strip()

def url_safe_name(name: str) -> str:
    """Convert display name to URL-safe format."""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

# Import calendar generation from main app
# We need to access the functions from app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import generate_ics_for_resident, PALETTE, norm_label
    HAS_APP = True
except ImportError:
    HAS_APP = False

if resident_param and HAS_APP:
    # Direct calendar request - serve ICS content
    resident_name = normalize_resident_name(resident_param)
    dailies = st.session_state.get("dailies", {})
    schedule_df = st.session_state.get("schedule_df", None)

    if dailies and resident_name in get_resident_list():
        # Generate ICS content
        ics_bytes = generate_ics_for_resident(resident_name, dailies, schedule_df)

        st.title(f"📅 {resident_name}'s Calendar")
        st.success("✅ Calendar feed is active!")

        # Show last updated time
        st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Download button
        safe_name = url_safe_name(resident_name)
        st.download_button(
            "⬇️ Download Calendar (.ics)",
            data=ics_bytes,
            file_name=f"{safe_name}_schedule.ics",
            mime="text/calendar",
            use_container_width=True
        )

        # Show raw ICS for manual copy (some apps need this)
        with st.expander("📋 Raw Calendar Data (for manual subscription)"):
            st.code(ics_bytes.decode('utf-8'), language=None)
            st.caption("Copy this entire text if your calendar app needs raw ICS data.")

        # Subscription instructions
        st.markdown("---")
        st.markdown("### 📱 How to Subscribe")

        # Get the current URL for subscription
        try:
            # This gets the base URL of the app
            app_url = st.session_state.get("_app_url", "https://your-app.streamlit.app")
        except:
            app_url = "https://your-app.streamlit.app"

        sub_url = f"{app_url}/Calendar_Feed?resident={safe_name}&download=ics"

        st.markdown(f"""
**Your subscription URL:**
```
{sub_url}
```

**Google Calendar:**
1. Open [Google Calendar](https://calendar.google.com)
2. Click the **+** next to "Other calendars" → **From URL**
3. Paste the URL above
4. Click **Add calendar**

**Note:** Google Calendar refreshes subscribed calendars every 12-24 hours.

**Outlook:**
1. Go to Calendar → **Add calendar** → **Subscribe from web**
2. Paste the URL above
3. Click **Import**

**Apple Calendar:**
1. File → **New Calendar Subscription**
2. Paste the URL above
3. Click **Subscribe**
        """)

    else:
        st.error(f"❌ Resident '{resident_name}' not found.")
        st.info("Make sure a schedule has been generated in the main app first.")

        available = get_resident_list()
        if available:
            st.markdown("**Available residents:**")
            for r in available:
                st.write(f"- {r}")

else:
    # No specific resident requested - show selection UI
    st.title("📅 Calendar Subscription")

    dailies = st.session_state.get("dailies", {})

    if not dailies:
        st.warning("⚠️ No schedule has been generated yet.")
        st.info("""
**To use calendar subscriptions:**
1. Go to the main **Yearly** tab
2. Enter your roster information
3. Click **Generate schedule from roster**
4. Come back here to get subscription URLs
        """)
        st.page_link("app.py", label="← Go to Main App", icon="🏠")
    else:
        st.success("✅ Schedule is available!")

        residents = get_resident_list()

        if residents:
            st.markdown("### Select a Resident")

            selected = st.selectbox(
                "Choose resident to get calendar URL:",
                options=residents,
                key="calendar_resident_select"
            )

            if selected:
                safe_name = url_safe_name(selected)

                # Create the subscription URL
                st.markdown("### 📎 Subscription URL")
                sub_url = f"?resident={safe_name}"

                st.code(sub_url, language=None)

                col1, col2 = st.columns(2)
                with col1:
                    # Link to the calendar page with params
                    st.link_button(
                        "🔗 Open Calendar Page",
                        f"/Calendar_Feed?resident={safe_name}",
                        use_container_width=True
                    )

                with col2:
                    # Direct download
                    if HAS_APP:
                        schedule_df = st.session_state.get("schedule_df", None)
                        ics_bytes = generate_ics_for_resident(selected, dailies, schedule_df)
                        st.download_button(
                            "⬇️ Download .ics",
                            data=ics_bytes,
                            file_name=f"{safe_name}_schedule.ics",
                            mime="text/calendar",
                            use_container_width=True
                        )

                st.markdown("---")
                st.markdown("""
### ℹ️ About Calendar Subscriptions

**How it works:**
1. Residents add the subscription URL to their calendar app
2. The calendar app periodically checks the URL for updates
3. When you update the schedule, residents see changes automatically

**Update frequency:**
- Google Calendar: Every 12-24 hours
- Outlook: Every 3-12 hours
- Apple Calendar: Configurable (default: every week)

**Tip:** For immediate updates, residents can manually refresh or re-download the .ics file.
                """)
        else:
            st.warning("No residents found in the schedule.")
