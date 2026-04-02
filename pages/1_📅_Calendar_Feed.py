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
import pickle
from datetime import date, timedelta
from pathlib import Path

st.set_page_config(page_title="Calendar Feed", page_icon="📅", layout="centered")

# --------------------------
# Load schedule from cache file
# --------------------------
SCHEDULE_CACHE_FILE = Path(__file__).parent.parent / ".schedule_cache.pkl"

def load_schedule_from_cache():
    """Load schedule data from file."""
    try:
        if SCHEDULE_CACHE_FILE.exists():
            with open(SCHEDULE_CACHE_FILE, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading schedule: {e}")
    return None

def get_cached_dailies():
    """Get dailies from cache or session state."""
    # First try session state
    if st.session_state.get("dailies"):
        return st.session_state.get("dailies"), st.session_state.get("schedule_df")

    # Fall back to cache file
    cached = load_schedule_from_cache()
    if cached:
        return cached.get("dailies", {}), cached.get("schedule_df")

    return {}, None

def get_resident_list(dailies):
    """Get list of residents from dailies."""
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
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import generate_ics_for_resident, PALETTE, norm_label
    HAS_APP = True
except ImportError as e:
    HAS_APP = False
    st.error(f"Could not import from app: {e}")

# Get query parameters
params = st.query_params
resident_param = params.get("resident", None)

# Load schedule data
dailies, schedule_df = get_cached_dailies()
residents = get_resident_list(dailies)

if resident_param and HAS_APP:
    # Direct calendar request - serve ICS content
    resident_name = normalize_resident_name(resident_param)

    # Try to find resident with case-insensitive match
    matched_resident = None
    for r in residents:
        if url_safe_name(r).lower() == resident_param.lower() or r.lower() == resident_name.lower():
            matched_resident = r
            break

    if matched_resident and dailies:
        # Generate ICS content
        ics_bytes = generate_ics_for_resident(matched_resident, dailies, schedule_df)

        st.title(f"📅 {matched_resident}'s Calendar")
        st.success("✅ Calendar feed is active!")

        # Show last updated time
        cached = load_schedule_from_cache()
        if cached and "saved_at" in cached:
            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cached["saved_at"]))
            st.caption(f"Schedule last updated: {saved_time}")

        # Download button
        safe_name = url_safe_name(matched_resident)
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

        st.markdown(f"""
**Your subscription URL path:**
```
/Calendar_Feed?resident={safe_name}
```

**Full URL example:**
```
https://your-app.streamlit.app/Calendar_Feed?resident={safe_name}
```

**Google Calendar:**
1. Open [Google Calendar](https://calendar.google.com)
2. Click the **+** next to "Other calendars" → **From URL**
3. Paste your full subscription URL
4. Click **Add calendar**

**Note:** Google Calendar refreshes subscribed calendars every 12-24 hours.

**Outlook:**
1. Go to Calendar → **Add calendar** → **Subscribe from web**
2. Paste your full subscription URL
3. Click **Import**

**Apple Calendar:**
1. File → **New Calendar Subscription**
2. Paste your full subscription URL
3. Click **Subscribe**
        """)

    else:
        st.error(f"❌ Resident '{resident_name}' not found.")

        if not dailies:
            st.warning("⚠️ No schedule has been saved yet.")
            st.info("""
**To fix this:**
1. Go to the main app (Yearly tab)
2. Generate or update the schedule
3. Come back here

The schedule needs to be generated at least once before calendars are available.
            """)
        else:
            st.info("Available residents:")
            for r in residents:
                safe = url_safe_name(r)
                st.write(f"- {r} → `?resident={safe}`")

else:
    # No specific resident requested - show selection UI
    st.title("📅 Calendar Subscription")

    if not dailies:
        st.warning("⚠️ No schedule has been generated yet.")
        st.info("""
**To use calendar subscriptions:**
1. Go to the main **Yearly** tab
2. Enter your roster information
3. Click **Generate schedule from roster**
4. Come back here to get subscription URLs
        """)
    else:
        st.success(f"✅ Schedule is available with {len(residents)} residents!")

        # Show last updated time
        cached = load_schedule_from_cache()
        if cached and "saved_at" in cached:
            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cached["saved_at"]))
            st.caption(f"Schedule last updated: {saved_time}")

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
                st.code(f"?resident={safe_name}", language=None)

                col1, col2 = st.columns(2)
                with col1:
                    st.link_button(
                        "🔗 Open Calendar Page",
                        f"/Calendar_Feed?resident={safe_name}",
                        use_container_width=True
                    )

                with col2:
                    if HAS_APP:
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
