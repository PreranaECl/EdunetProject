"""
app.py — Emergency Assist AI — Streamlit Application Entry Point.

Implements the full user journey:
  Step 1  → Display emergency contact prominently
  Step 2  → Accept API configuration in sidebar
  Step 3  → Accept emergency input (category, description, location, image)
  Step 4  → Run two-agent Replicate workflow
  Step 5  → Display results (assessment, guidance, map, help centers)
  Step 6  → Download emergency assistance report
  Step 7  → Show performance dashboard

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import streamlit as st

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Emergency Assist AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports (after page config) ──────────────────────────────────────
import agents
import groq_client as replicate_client  # drop-in replacement
import guardrails
import maps_service
import memory
import utils
from streamlit_js_eval import streamlit_js_eval
from utils import (
    average_response_time,
    build_emergency_report_text,
    category_icon,
    check_rate_limit,
    check_token_budget,
    redact_pii,
    severity_color,
    update_metrics_from_workflow,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
)
from schemas import AgentState, SessionMetrics

# ── Initialise cross-session SQLite DB ───────────────────────────────────────
memory.init_db()

# ── Constants ─────────────────────────────────────────────────────────────────
EMERGENCY_CATEGORIES = [
    "Medical Emergency",
    "Road Accident",
    "Fire Emergency",
    "Crime or Security Threat",
    "Natural Disaster",
    "Flood or Water Emergency",
    "Building Collapse",
    "Missing Person",
    "Other Emergency",
]

SEVERITY_BADGE = {
    "Critical": "🔴 Critical",
    "High":     "🟠 High",
    "Moderate": "🟡 Moderate",
    "Low":      "🟢 Low",
    "Unknown":  "⚪ Unknown",
}


# ── Session state initialisation ──────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "metrics": utils.init_session_metrics(),
        "workflow_result": None,
        "last_emergency_type": None,
        "last_description": None,
        "last_location": None,
        "replicate_configured": False,
        "maps_configured": False,
        "show_agent_details": False,
        "show_performance": False,
        # Live location state
        "live_lat": "",
        "live_lon": "",
        "live_address": "",
        "token_budget": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# =============================================================================
#  SIDEBAR
# =============================================================================

def render_sidebar() -> dict:
    """Render the configuration sidebar and return active settings."""
    with st.sidebar:
        st.markdown("## 🔧 Configuration")

        # ── Groq AI ───────────────────────────────────────────────────────────
        st.markdown("### 🤖 Groq AI (Free)")
        st.caption("Get a free API key at https://console.groq.com — no payment needed.")

        # Load from Streamlit secrets or environment variables as defaults
        try:
            default_replicate_token = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            default_replicate_token = ""
        default_replicate_token = (
            default_replicate_token
            or os.environ.get("GROQ_API_KEY", "")
            or os.environ.get("REPLICATE_API_TOKEN", "")
        )

        replicate_token = st.text_input(
            "Groq API Key",
            value=default_replicate_token,
            type="password",
            help="Your Groq API key (gsk_...). Free — no payment required. Never shared or logged.",
            placeholder="gsk_...",
        )

        assessment_model = st.text_input(
            "Assessment Model",
            value="llama-3.2-11b-vision-preview",
            help="Groq model for Emergency Assessment Agent. llama-3.2-11b-vision-preview supports images.",
        )

        response_model = st.text_input(
            "Response Model",
            value="llama-3.3-70b-versatile",
            help="Groq model for Emergency Response Agent.",
        )

        # ── Map Service (free — no key needed) ───────────────────────────────
        st.markdown("### 🗺️ Map Service")
        st.info(
            "✅ **Free OpenStreetMap** is used for geocoding and nearby search.  \n"
            "No API key required. Powered by Nominatim + Overpass API."
        )
        # Always configure and mark as ready
        maps_service.configure_maps()
        st.session_state["maps_configured"] = True

        # ── Connection Status ─────────────────────────────────────────────────
        st.markdown("### 📡 Connection Status")

        col1, col2 = st.columns(2)

        # Groq connection
        with col1:
            if replicate_token:
                try:
                    replicate_client.configure_replicate(
                        api_token=replicate_token,
                        assessment_model=assessment_model,
                        response_model=response_model,
                    )
                    st.session_state["replicate_configured"] = True
                    st.success("✅ Groq AI")
                except Exception as exc:
                    st.session_state["replicate_configured"] = False
                    st.error(f"❌ Groq AI\n{exc}")
            else:
                st.session_state["replicate_configured"] = False
                st.warning("⚠️ Groq AI\nNot configured")

        # Free map service status
        with col2:
            ok, msg = maps_service.validate_maps_connection()
            if ok:
                st.session_state["maps_configured"] = True
                st.success(f"✅ OSM Maps")
            else:
                st.session_state["maps_configured"] = False
                st.warning(f"⚠️ OSM Maps\n{msg}")

        # ── App Settings ──────────────────────────────────────────────────────
        st.markdown("### ⚙️ Application Settings")
        st.session_state["show_agent_details"] = st.toggle(
            "Show agent details", value=False
        )
        st.session_state["show_performance"] = st.toggle(
            "Show performance metrics", value=True
        )

        # Token budget
        st.session_state["token_budget"] = st.number_input(
            "Token budget (0 = unlimited)",
            min_value=0,
            max_value=500_000,
            value=st.session_state.get("token_budget", 0),
            step=1000,
            help="Stop processing new requests once this many tokens have been used in the session.",
        )

        # Rate limit display
        st.caption(
            f"🚦 Rate limit: {RATE_LIMIT_REQUESTS_PER_MINUTE} requests / minute per session."
        )

        # Memory controls
        st.markdown("### 🧠 Session Memory")
        metrics: SessionMetrics = st.session_state.get("metrics", utils.init_session_metrics())
        st.caption(f"Session requests: {metrics.total_requests}  |  "
                   f"Tokens: {metrics.input_tokens + metrics.output_tokens:,}  |  "
                   f"Est. cost: ${metrics.total_cost_usd:.4f}")
        if st.button("🗑️ Clear session metrics", use_container_width=True):
            st.session_state["metrics"] = utils.init_session_metrics()
            st.session_state["workflow_result"] = None
            st.rerun()
        if memory.db_exists():
            cumulative = memory.load_cumulative_stats()
            st.caption(
                f"Lifetime: {cumulative.get('total_requests', 0)} requests  |  "
                f"${cumulative.get('total_cost_usd', 0.0):.4f} total"
            )
            if st.button("🗑️ Clear saved history", use_container_width=True):
                memory.clear_history()
                st.success("History cleared.")

        st.markdown("---")
        st.caption(
            "📢 Location is sent to OpenStreetMap (Nominatim) for geocoding. "
            "Nearby search uses the free Overpass API. "
            "Emergency descriptions and images are sent to Replicate AI for analysis. "
            "Do not include unnecessary personal information."
        )

    return {
        "replicate_token": replicate_token,
        "assessment_model": assessment_model,
        "response_model": response_model,
    }


# =============================================================================
#  EMERGENCY CONTACT BANNER
# =============================================================================

def render_emergency_banner() -> None:
    st.error(
        "🚨 **If someone is in IMMEDIATE DANGER — Call 112 (India) RIGHT NOW.**  \n"
        "Do NOT wait for AI analysis. **112 connects you to Police · Fire · Ambulance · Disaster Relief.**"
    )


# =============================================================================
#  MAIN INPUT FORM
# =============================================================================

def _request_live_location() -> None:
    """
    Render the live location capture widget (outside the form).
    Uses streamlit-js-eval to call navigator.geolocation in the browser.
    Writes lat/lon + address into st.session_state on success.
    """
    # ── Live location button ──────────────────────────────────────────────────
    loc_col1, loc_col2 = st.columns([2, 1])
    with loc_col1:
        st.markdown(
            "📍 **Live Location** — click the button to auto-detect your current location "
            "using your device's GPS/network. Browser permission is required."
        )
    with loc_col2:
        get_loc = st.button(
            "📍 Use My Live Location",
            key="btn_live_loc",
            use_container_width=True,
            help="Requests your browser's location. Your device must allow location access.",
        )

    if get_loc:
        # Ask the browser for geolocation via JS
        geo_result = streamlit_js_eval(
            js_expressions="""
                new Promise((resolve, reject) => {
                    if (!navigator.geolocation) {
                        resolve({error: "Geolocation is not supported by this browser."});
                        return;
                    }
                    navigator.geolocation.getCurrentPosition(
                        (pos) => resolve({
                            lat: pos.coords.latitude,
                            lon: pos.coords.longitude,
                            accuracy: pos.coords.accuracy
                        }),
                        (err) => resolve({error: err.message}),
                        {enableHighAccuracy: true, timeout: 10000, maximumAge: 0}
                    );
                })
            """,
            key="geo_js",
        )

        if geo_result:
            if "error" in geo_result:
                st.error(
                    f"❌ Location access failed: {geo_result['error']}  \n"
                    "Please allow location permission in your browser, or enter your location manually below."
                )
            else:
                lat = geo_result.get("lat")
                lon = geo_result.get("lon")
                accuracy = geo_result.get("accuracy", 0)
                if lat is not None and lon is not None:
                    # Validate coordinate ranges
                    from guardrails import validate_coordinates
                    coord_check = validate_coordinates(lat, lon)
                    if coord_check.passed:
                        # Reverse-geocode to get a readable address
                        address = maps_service.reverse_geocode(lat, lon)
                        st.session_state["live_lat"] = str(round(lat, 6))
                        st.session_state["live_lon"] = str(round(lon, 6))
                        st.session_state["live_address"] = address
                        acc_str = f"{accuracy:.0f} m" if accuracy else "unknown"
                        st.success(
                            f"✅ **Live location captured!** ({round(lat,5)}, {round(lon,5)})  \n"
                            f"📍 {address}  \n"
                            f"Accuracy: ~{acc_str}"
                        )
                    else:
                        st.error(f"Invalid coordinates received: {coord_check.message}")

    # Show current live location status
    if st.session_state.get("live_lat") and st.session_state.get("live_lon"):
        st.info(
            f"🟢 Live location active: **{st.session_state.get('live_address', '')}**  \n"
            f"Coordinates: {st.session_state['live_lat']}, {st.session_state['live_lon']}  \n"
            "_These will be used automatically in the form below._"
        )
        if st.button("🗑️ Clear live location", key="btn_clear_loc"):
            st.session_state.pop("live_lat", None)
            st.session_state.pop("live_lon", None)
            st.session_state.pop("live_address", None)
            st.rerun()


def render_input_form() -> Optional[dict]:
    """
    Render the emergency input form.
    Returns form data dict if submitted, else None.
    """
    st.markdown("## 📋 Report an Emergency")
    st.caption(
        "Fill in the details below. All fields except image are required to receive AI assistance."
    )

    # Live location capture (outside the form — JS must run outside st.form)
    _request_live_location()
    st.markdown("")

    # Pre-populate from live location if available
    _live_lat = st.session_state.get("live_lat", "")
    _live_lon = st.session_state.get("live_lon", "")
    _live_addr = st.session_state.get("live_address", "")

    with st.form("emergency_form", clear_on_submit=False):
        col_left, col_right = st.columns([1, 1])

        with col_left:
            emergency_type = st.selectbox(
                "1️⃣ Emergency Category",
                options=EMERGENCY_CATEGORIES,
                help="Select the category that best describes the emergency.",
            )

            location_input = st.text_input(
                "2️⃣ Your Current Location",
                value=_live_addr,                     # pre-filled from live location
                placeholder="e.g. MG Road, Bengaluru, Karnataka",
                help=(
                    "Auto-filled from live location, or enter your area/landmark manually. "
                    "Used to find nearby emergency services."
                ),
            )

            lat_col, lon_col = st.columns(2)
            with lat_col:
                lat_manual = st.text_input(
                    "Latitude",
                    value=_live_lat,                  # pre-filled from live GPS
                    placeholder="e.g. 12.9716",
                    help="Auto-filled from live location, or enter manually.",
                )
            with lon_col:
                lon_manual = st.text_input(
                    "Longitude",
                    value=_live_lon,                  # pre-filled from live GPS
                    placeholder="e.g. 77.5946",
                    help="Auto-filled from live location, or enter manually.",
                )

        with col_right:
            description = st.text_area(
                "3️⃣ Describe the Emergency",
                height=160,
                placeholder=(
                    "Describe what has happened. Include relevant details such as "
                    "number of people involved, visible injuries, location hazards, etc."
                ),
                help=f"Maximum {guardrails.MAX_DESCRIPTION_CHARS} characters.",
            )

            uploaded_file = st.file_uploader(
                "4️⃣ Upload Emergency Image (Optional)",
                type=["jpg", "jpeg", "png"],
                help="A photo may help identify visible hazards. JPG/PNG, max 5 MB.",
            )

        submitted = st.form_submit_button(
            "🔍 Find Emergency Assistance",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    return {
        "emergency_type": emergency_type,
        "description": description,
        "location_input": location_input,
        "lat_manual": lat_manual,
        "lon_manual": lon_manual,
        "uploaded_file": uploaded_file,
    }


# =============================================================================
#  INPUT VALIDATION
# =============================================================================

def validate_inputs(form_data: dict) -> tuple[bool, list[str], int]:
    """
    Run all input guardrails. Returns (all_valid, error_messages, intervention_count).
    """
    errors = []
    interventions = 0

    # Description
    desc_result = guardrails.validate_text_input(
        form_data["description"], field_name="Emergency description"
    )
    if not desc_result:
        errors.append(desc_result.message)
        if desc_result.intervention:
            interventions += 1

    # PII check
    has_pii, pii_types = guardrails.check_pii(form_data["description"])
    if has_pii:
        st.warning(
            f"⚠️ Your description may contain sensitive information "
            f"({', '.join(pii_types)}). Please remove unnecessary personal details before submitting."
        )
        interventions += 1

    # Location
    loc = form_data.get("location_input", "").strip()
    lat_str = form_data.get("lat_manual", "").strip()
    lon_str = form_data.get("lon_manual", "").strip()

    if lat_str and lon_str:
        coord_result = guardrails.validate_coordinates(lat_str, lon_str)
        if not coord_result:
            errors.append(coord_result.message)
            if coord_result.intervention:
                interventions += 1
    elif not loc:
        errors.append(
            "Please enter your current location (area, locality, or landmark) "
            "so we can find nearby emergency services."
        )

    if loc:
        loc_result = guardrails.validate_location_text(loc)
        if not loc_result:
            errors.append(loc_result.message)
            if loc_result.intervention:
                interventions += 1

    # Image
    uploaded = form_data.get("uploaded_file")
    if uploaded is not None:
        file_bytes = uploaded.read()
        uploaded.seek(0)
        img_result = guardrails.validate_image(
            file_bytes, uploaded.name
        )
        if not img_result:
            errors.append(img_result.message)
            if img_result.intervention:
                interventions += 1

    return (len(errors) == 0, errors, interventions)


# =============================================================================
#  RESULTS RENDERING
# =============================================================================

def render_assessment(assessment) -> None:
    severity = assessment.severity_level
    badge = SEVERITY_BADGE.get(severity, severity)
    color = severity_color(severity)

    st.markdown(f"### {badge} Emergency Assessment")
    st.markdown(
        f"<div style='border-left:4px solid {color}; padding:10px; border-radius:4px;'>"
        f"<b>{assessment.emergency_summary}</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        if assessment.observed_indicators:
            st.markdown("**Observed Indicators**")
            for item in assessment.observed_indicators:
                st.markdown(f"- {item}")
    with col2:
        if assessment.potential_risks:
            st.markdown("**Potential Risks**")
            for item in assessment.potential_risks:
                st.markdown(f"- {item}")

    if assessment.information_gaps:
        st.info("ℹ️ **Information gaps:** " + " • ".join(assessment.information_gaps))

    if assessment.immediate_emergency_contact_required:
        st.error("🚨 **Call 112 immediately.** This situation may be life-threatening.")


def render_response(response) -> None:
    st.markdown("### 🛡️ Emergency Safety Guidance")
    st.markdown(response.emergency_summary)

    if response.immediate_actions:
        st.markdown("**✅ Immediate Safety Actions**")
        for action in response.immediate_actions:
            st.markdown(f"- {action}")

    if response.safety_warnings:
        st.warning("⚠️ **Safety Warnings**\n\n" + "\n".join(f"- {w}" for w in response.safety_warnings))

    if response.recommended_services:
        st.markdown("**🚑 Recommended Emergency Services:** " + " · ".join(response.recommended_services))

    st.error("📞 **Emergency Contact: 112** — India Emergency Response Support")
    st.caption(response.disclaimer)


def render_help_centers(help_centers: list, user_lat=None, user_lon=None) -> None:
    if not help_centers:
        st.warning(
            "No verified nearby help centers were found. "
            "Please call **112** for emergency assistance."
        )
        return

    st.markdown(f"### 📍 {len(help_centers)} Nearby Help Center(s) Found")
    st.caption("Distances are straight-line (haversine). Actual travel distance may differ.")

    for c in help_centers:
        dist_str = f"{c['distance_km']} km away" if c.get("distance_km") is not None else "distance unknown"
        phone_str = c.get("phone") or "—"

        with st.expander(f"📌 {c['name']}  [{c['category']}]  — {dist_str}", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Address:** {c.get('address', 'N/A')}")
                st.markdown(f"**Category:** {c['category']}")
                st.markdown(f"**Distance:** {dist_str}")
                st.markdown(f"**Phone:** {phone_str}")
            with col2:
                if c.get("maps_url"):
                    st.markdown(
                        f'<a href="{c["maps_url"]}" target="_blank">'
                        '🗺️ Open in Google Maps</a>',
                        unsafe_allow_html=True,
                    )


# Category → Folium colour and icon
_CATEGORY_STYLE = {
    "Hospital":             ("blue",      "plus-sign"),
    "Ambulance Service":    ("lightblue", "ambulance"),
    "Police Station":       ("darkblue",  "flag"),
    "Fire Station":         ("orange",    "fire"),
    "Fire & Rescue":        ("orange",    "fire"),
    "Disaster Relief Center": ("purple",  "home"),
    "Emergency Shelter":    ("green",     "home"),
}


def render_map(help_centers: list, user_lat: float, user_lon: float) -> None:
    """
    Render a live interactive map with:
    - Switchable tile layers (Street / Satellite / Terrain / Dark)
    - Distance rings (1 km and 3 km)
    - Per-category feature groups with layer control
    - Routing polylines from user location to each help center
    - Fullscreen plugin
    - Embedded Google Maps tab for live street-level view
    """
    try:
        import folium
        from folium import plugins as folium_plugins
        from streamlit_folium import st_folium

        map_tab, gmaps_tab = st.tabs(["🗺️ Interactive Map", "🌐 Live Google Maps"])

        with map_tab:
            # ── Base map ──────────────────────────────────────────────────────
            m = folium.Map(
                location=[user_lat, user_lon],
                zoom_start=14,
                tiles=None,           # we add tiles manually so names appear in LayerControl
            )

            # ── Tile layers ───────────────────────────────────────────────────
            folium.TileLayer(
                tiles="OpenStreetMap",
                name="Street Map",
                control=True,
            ).add_to(m)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                name="Satellite",
                control=True,
            ).add_to(m)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Topo",
                name="Terrain",
                control=True,
            ).add_to(m)

            folium.TileLayer(
                tiles="CartoDB dark_matter",
                name="Dark Mode",
                control=True,
            ).add_to(m)

            # ── Distance rings ────────────────────────────────────────────────
            rings_group = folium.FeatureGroup(name="Distance Rings", show=True)
            for radius_m, label in [(1000, "1 km"), (3000, "3 km"), (5000, "5 km")]:
                folium.Circle(
                    location=[user_lat, user_lon],
                    radius=radius_m,
                    color="#3b82d4",
                    weight=1,
                    fill=True,
                    fill_opacity=0.04,
                    tooltip=label,
                ).add_to(rings_group)
            rings_group.add_to(m)

            # ── User location marker ──────────────────────────────────────────
            user_group = folium.FeatureGroup(name="Your Location", show=True)
            folium.Marker(
                location=[user_lat, user_lon],
                popup=folium.Popup(
                    "<b>📍 Your Location</b><br>Emergency reported here.",
                    max_width=200,
                ),
                tooltip="📍 Your Location",
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
            ).add_to(user_group)
            user_group.add_to(m)

            # ── Help center markers + routing lines ───────────────────────────
            # Build one FeatureGroup per category so each can be toggled
            category_groups: dict = {}

            for c in help_centers:
                if not (c.get("latitude") and c.get("longitude")):
                    continue

                cat = c["category"]
                if cat not in category_groups:
                    category_groups[cat] = folium.FeatureGroup(
                        name=f"🏥 {cat}" if "Hospital" in cat
                        else f"🚔 {cat}" if "Police" in cat
                        else f"🔥 {cat}" if "Fire" in cat
                        else f"🚑 {cat}" if "Ambulance" in cat
                        else f"🏚️ {cat}",
                        show=True,
                    )

                colour, icon_name = _CATEGORY_STYLE.get(cat, ("gray", "info-sign"))
                dist_str = f"{c['distance_km']} km away" if c.get("distance_km") is not None else "distance unknown"
                phone_str = c.get("phone") or "—"

                popup_html = (
                    f"<div style='min-width:220px'>"
                    f"<b style='font-size:14px'>{c['name']}</b><br>"
                    f"<span style='color:#555'>{cat}</span><br><br>"
                    f"📍 {c.get('address', 'N/A')}<br>"
                    f"📏 {dist_str}<br>"
                    f"📞 {phone_str}<br>"
                )
                if c.get("maps_url"):
                    popup_html += (
                        f"<br><a href='{c['maps_url']}' target='_blank' "
                        f"style='color:#3b82d4'>🗺️ Open in Google Maps</a>"
                    )
                popup_html += "</div>"

                folium.Marker(
                    location=[c["latitude"], c["longitude"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{c['name']} — {dist_str}",
                    icon=folium.Icon(color=colour, icon=icon_name, prefix="glyphicon"),
                ).add_to(category_groups[cat])

                # Routing polyline (straight-line — labeled as approximate)
                folium.PolyLine(
                    locations=[
                        [user_lat, user_lon],
                        [c["latitude"], c["longitude"]],
                    ],
                    color="#dc2626" if c.get("distance_km", 99) < 2 else "#f97316",
                    weight=2,
                    opacity=0.55,
                    dash_array="6 4",
                    tooltip=f"~{dist_str} (straight-line)",
                ).add_to(category_groups[cat])

            for grp in category_groups.values():
                grp.add_to(m)

            # ── Fullscreen plugin ─────────────────────────────────────────────
            try:
                folium_plugins.Fullscreen(
                    position="topright",
                    title="Fullscreen",
                    title_cancel="Exit fullscreen",
                ).add_to(m)
            except Exception:
                pass  # plugin may not be available in all folium versions

            # ── Layer control ─────────────────────────────────────────────────
            folium.LayerControl(position="topright", collapsed=False).add_to(m)

            # ── Fit map to all markers ────────────────────────────────────────
            all_lats = [user_lat] + [c["latitude"] for c in help_centers if c.get("latitude")]
            all_lons = [user_lon] + [c["longitude"] for c in help_centers if c.get("longitude")]
            if len(all_lats) > 1:
                m.fit_bounds(
                    [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]],
                    max_zoom=15,
                )

            st_folium(m, width="100%", height=520, returned_objects=[])

        # ── Live OpenStreetMap embed tab ──────────────────────────────────────
        with gmaps_tab:
            st.caption(
                "Live OpenStreetMap view centred on your reported location — free, no API key required. "
                "Click any help center link above to open directions."
            )
            # Free OSM embed via openstreetmap.org iframe — no key needed
            osm_embed_url = (
                f"https://www.openstreetmap.org/export/embed.html"
                f"?bbox={user_lon-0.05},{user_lat-0.05},{user_lon+0.05},{user_lat+0.05}"
                f"&layer=mapnik"
                f"&marker={user_lat},{user_lon}"
            )
            osm_full_url = (
                f"https://www.openstreetmap.org/?mlat={user_lat}&mlon={user_lon}"
                f"&zoom=15#map=15/{user_lat}/{user_lon}"
            )
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:8px; overflow:hidden;">
                  <iframe
                    src="{osm_embed_url}"
                    width="100%" height="500"
                    style="border:0; display:block;"
                    allowfullscreen
                    loading="lazy">
                  </iframe>
                </div>
                <p style="margin-top:8px; font-size:12px; color:#57606a;">
                  📍 Map centred on ({user_lat:.5f}, {user_lon:.5f}) — OpenStreetMap contributors.
                  &nbsp;
                  <a href="{osm_full_url}" target="_blank" style="color:#3b82d4;">
                    Open full OpenStreetMap ↗
                  </a>
                </p>
                """,
                unsafe_allow_html=True,
            )

    except ImportError:
        st.info(
            "Map visualisation requires the `folium` and `streamlit-folium` packages. "
            "Run `pip install folium streamlit-folium` then restart the app."
        )
    except Exception as exc:
        st.warning(f"Map could not be rendered: {exc}")


# =============================================================================
#  PERFORMANCE DASHBOARD TAB
# =============================================================================

# Agent state → badge
_STATE_BADGE = {
    "idle":      "⚪ Idle",
    "pending":   "🔵 Pending",
    "running":   "🟡 Running",
    "retrying":  "🟠 Retrying",
    "completed": "✅ Completed",
    "failed":    "❌ Failed",
    "blocked":   "🚫 Blocked",
}


def render_performance_dashboard(metrics: SessionMetrics) -> None:
    import pandas as pd

    st.markdown("## 📊 Agent Performance Dashboard")

    dash_tab, trend_tab, state_tab, history_tab = st.tabs(
        ["📈 Summary", "📉 Trends", "🤖 Agent States", "🗄️ History"]
    )

    # ── Summary tab ───────────────────────────────────────────────────────────
    with dash_tab:
        st.caption("Metrics reflect the current application session.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Requests", metrics.total_requests)
        col2.metric("Successful", metrics.successful_workflows)
        col3.metric("Failed", metrics.failed_workflows)
        col4.metric("Guardrail Interventions", metrics.guardrail_interventions)

        col5, col6, col7, col8 = st.columns(4)
        avg_time = average_response_time(metrics)
        col5.metric("Avg Response Time", f"{avg_time}s")
        col6.metric("Replicate API Calls", metrics.gemini_api_calls)
        col7.metric("Maps API Calls", metrics.maps_api_calls)
        col8.metric("Tokens (in / out)", f"{metrics.input_tokens:,} / {metrics.output_tokens:,}")

        # Cost (Fix 2)
        if metrics.total_cost_usd > 0:
            st.success(
                f"💰 **Estimated session cost: ${metrics.total_cost_usd:.4f} USD** "
                f"(approximate — based on token counts × per-model pricing)"
            )
        else:
            st.info("💰 Estimated cost: $0.00 — no completed workflows yet this session.")

        st.markdown("### ⏱️ Component Latency (Last Workflow)")
        lat_col1, lat_col2, lat_col3 = st.columns(3)
        lat_col1.metric("Assessment Agent", f"{metrics.assessment_latency_s:.2f}s")
        lat_col2.metric("Response Agent", f"{metrics.response_latency_s:.2f}s")
        lat_col3.metric("Google Maps", f"{metrics.maps_latency_s:.2f}s")

        # Execution log table
        if metrics.execution_log:
            st.markdown("### 📋 Agent Execution Log")
            log_data = [
                {
                    "Component": r.component,
                    "State": _STATE_BADGE.get(r.state.value if hasattr(r.state, "value") else r.state, r.status),
                    "Status": r.status,
                    "Time (s)": f"{r.execution_time_s:.3f}",
                    "Validation": r.validation,
                    "Tokens In": r.input_tokens,
                    "Tokens Out": r.output_tokens,
                    "Est. Cost": f"${r.estimated_cost_usd:.6f}" if r.estimated_cost_usd else "—",
                    "Notes": r.notes[:80] if r.notes else "",
                }
                for r in metrics.execution_log
            ]
            st.dataframe(log_data, use_container_width=True)
        else:
            st.info("No workflow executions recorded yet.")

    # ── Trends tab (Fix 7) ────────────────────────────────────────────────────
    with trend_tab:
        st.markdown("### 📉 Latency & Token Trends")
        if not metrics.latency_history:
            st.info("No workflow data yet. Run at least one emergency request to see trends.")
        else:
            history = metrics.latency_history

            # Latency line chart
            latency_df = {
                "Request": list(range(1, len(history) + 1)),
                "Assessment (s)": [p.assessment_s for p in history],
                "Response (s)":   [p.response_s for p in history],
                "Maps (s)":       [p.maps_s for p in history],
                "Total (s)":      [p.total_s for p in history],
            }
            st.markdown("**Response Time per Workflow**")
            st.line_chart(
                {k: v for k, v in latency_df.items() if k != "Request"},
                use_container_width=True,
                height=260,
            )

            # Token usage bar chart
            st.markdown("**Token Usage per Workflow**")
            token_df = {
                "Input Tokens":  [p.input_tokens for p in history],
                "Output Tokens": [p.output_tokens for p in history],
            }
            st.bar_chart(token_df, use_container_width=True, height=220)

            # Cost trend
            st.markdown("**Estimated Cost per Workflow (USD)**")
            cost_data = {"Est. Cost ($)": [p.cost_usd for p in history]}
            st.line_chart(cost_data, use_container_width=True, height=180)

            # Tabular view
            with st.expander("View raw trend data"):
                rows = [
                    {
                        "#": i + 1,
                        "Timestamp": p.timestamp[:19].replace("T", " "),
                        "Type": p.emergency_type,
                        "Assessment (s)": p.assessment_s,
                        "Response (s)": p.response_s,
                        "Maps (s)": p.maps_s,
                        "Total (s)": p.total_s,
                        "Tokens In": p.input_tokens,
                        "Tokens Out": p.output_tokens,
                        "Cost ($)": f"{p.cost_usd:.6f}",
                    }
                    for i, p in enumerate(history)
                ]
                st.dataframe(rows, use_container_width=True)

    # ── Agent state machine tab (Fix 8) ──────────────────────────────────────
    with state_tab:
        st.markdown("### 🤖 Agent State Machine — Last Workflow")
        if not metrics.execution_log:
            st.info("No agent executions recorded yet.")
        else:
            # Show the last 3 records (one per workflow component)
            recent = metrics.execution_log[-3:]
            for rec in recent:
                state_val = rec.state.value if hasattr(rec.state, "value") else str(rec.state)
                badge = _STATE_BADGE.get(state_val, state_val)
                color = {
                    "completed": "#22c55e",
                    "failed":    "#dc2626",
                    "blocked":   "#7c3aed",
                    "running":   "#eab308",
                    "retrying":  "#f97316",
                    "pending":   "#3b82d4",
                }.get(state_val, "#6b7280")

                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding:10px 14px; "
                    f"margin-bottom:10px; border-radius:4px; background:#f7f8fa;'>"
                    f"<b>{rec.component}</b> &nbsp;|&nbsp; {badge}<br>"
                    f"<small>Time: {rec.execution_time_s:.3f}s &nbsp;·&nbsp; "
                    f"Validation: {rec.validation} &nbsp;·&nbsp; "
                    f"Tokens: {rec.input_tokens} in / {rec.output_tokens} out</small>"
                    f"{'<br><small style=color:#dc2626>' + rec.notes + '</small>' if rec.notes and state_val == 'failed' else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # State transition diagram as mermaid
            if len(recent) >= 2:
                st.markdown("**State Transition Flow**")
                states = " --> ".join(
                    f"{r.component.split()[0]}[{_STATE_BADGE.get(r.state.value if hasattr(r.state,'value') else str(r.state), r.status)}]"
                    for r in recent
                )
                st.code(f"graph LR\n    {states}", language="")

    # ── History tab (Fix 6) ───────────────────────────────────────────────────
    with history_tab:
        st.markdown("### 🗄️ Cross-Session Workflow History")
        rows = memory.load_workflow_history(limit=100)
        if not rows:
            st.info("No saved history yet. Complete a workflow to start recording.")
        else:
            st.caption(f"{len(rows)} most recent workflows stored in local SQLite database.")
            cumulative = memory.load_cumulative_stats()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lifetime Requests", cumulative.get("total_requests", 0))
            c2.metric("Lifetime Successful", cumulative.get("successful", 0))
            c3.metric("Lifetime Tokens", f"{cumulative.get('total_input_tokens', 0) + cumulative.get('total_output_tokens', 0):,}")
            c4.metric("Lifetime Cost", f"${cumulative.get('total_cost_usd', 0.0):.4f}")

            # Display table
            display_rows = [
                {
                    "Time": r["created_at"][:19].replace("T", " "),
                    "Emergency Type": r["emergency_type"],
                    "Severity": r["severity_level"],
                    "Status": "✅" if r["status"] == "success" else "❌",
                    "Total (s)": f"{r['total_time_s']:.2f}",
                    "Tokens": r["input_tokens"] + r["output_tokens"],
                    "Cost ($)": f"{r['cost_usd']:.6f}",
                    "Help Centers": r["maps_results"],
                    "Guardrails": r["guardrail_hits"],
                }
                for r in rows
            ]
            st.dataframe(display_rows, use_container_width=True, height=400)


# =============================================================================
#  MAIN APPLICATION
# =============================================================================

def main() -> None:
    # Header
    st.markdown("# 🚨 EMERGENCY ASSIST AI")
    st.markdown("### *AI-Powered Public Emergency Assistance and Nearest Help Center Finder*")
    st.caption(
        "Get emergency safety guidance and discover nearby hospitals, police stations, "
        "fire stations, ambulance services, and disaster relief centers."
    )

    # Always-visible emergency banner
    render_emergency_banner()

    # Sidebar configuration
    config = render_sidebar()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_assist, tab_dashboard = st.tabs(["🆘 Emergency Assistance", "📊 Agent Performance"])

    with tab_assist:
        _render_assistance_tab(config)

    with tab_dashboard:
        render_performance_dashboard(st.session_state["metrics"])
        # note: history/trends/state sub-tabs are rendered inside render_performance_dashboard


def _render_assistance_tab(config: dict) -> None:
    """Render the main emergency assistance tab."""

    # ── Input form ────────────────────────────────────────────────────────────
    form_data = render_input_form()

    # ── Display previous results if available ─────────────────────────────────
    if st.session_state.get("workflow_result") and form_data is None:
        _display_results(st.session_state["workflow_result"])
        return

    if form_data is None:
        # Show static quick-reference contacts
        st.markdown("---")
        st.markdown("### 📞 Emergency Contacts (India)")
        contact_cols = st.columns(4)
        contact_cols[0].info("**112**\nEmergency Response")
        contact_cols[1].info("**100**\nPolice")
        contact_cols[2].info("**101**\nFire")
        contact_cols[3].info("**102 / 108**\nAmbulance")
        return

    # ── Configuration check ───────────────────────────────────────────────────
    if not st.session_state.get("replicate_configured"):
        st.error(
            "❌ Groq API is not configured. "
            "Please enter a free Groq API key (gsk_...) in the sidebar.  \n"
            "Get one free at https://console.groq.com — no payment needed.\n\n"
            "🚨 While configuring, remember: **call 112 for immediate danger**."
        )
        return

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_ok, rate_msg = check_rate_limit(st.session_state["metrics"])
    if not rate_ok:
        st.error(f"🚦 {rate_msg}")
        st.error("🚨 **If this is an immediate emergency, call 112 now.**")
        return

    # ── Token budget ──────────────────────────────────────────────────────────
    token_budget = int(st.session_state.get("token_budget", 0))
    budget_ok, budget_msg = check_token_budget(st.session_state["metrics"], token_budget)
    if not budget_ok:
        st.error(f"💰 {budget_msg}")
        st.error("🚨 **If this is an immediate emergency, call 112 now.**")
        return
    if budget_msg:
        st.warning(budget_msg)

    # ── Input validation ──────────────────────────────────────────────────────
    valid, errors, intervention_count = validate_inputs(form_data)

    if intervention_count > 0:
        guardrails_metrics: SessionMetrics = st.session_state["metrics"]
        for _ in range(intervention_count):
            utils.record_guardrail_intervention(guardrails_metrics)

    if not valid:
        for err in errors:
            st.error(f"❌ {err}")
        st.error("🚨 **If this is an immediate emergency, call 112 now.**")
        return

    # ── PII redaction (Fix 3) — strip before sending to AI ───────────────────
    clean_description, redacted_types = redact_pii(form_data["description"])
    if redacted_types:
        st.warning(
            f"⚠️ Sensitive information ({', '.join(redacted_types)}) was detected and "
            f"automatically removed from your description before AI processing."
        )
        utils.record_guardrail_intervention(st.session_state["metrics"])

    # ── Prepare image ─────────────────────────────────────────────────────────
    image_bytes: Optional[bytes] = None
    image_mime = "image/jpeg"
    uploaded = form_data.get("uploaded_file")
    if uploaded is not None:
        raw_bytes = uploaded.read()
        try:
            image_bytes, image_mime = utils.prepare_image(raw_bytes)
            image_bytes = guardrails.strip_image_metadata(image_bytes)
        except Exception as exc:
            st.warning(f"Image processing issue: {exc}. Proceeding without image.")
            image_bytes = None

    # ── Determine effective location ──────────────────────────────────────────
    lat_str = form_data.get("lat_manual", "").strip()
    lon_str = form_data.get("lon_manual", "").strip()
    location_input = form_data["location_input"].strip()

    effective_location = location_input
    user_lat = user_lon = None

    if lat_str and lon_str:
        try:
            user_lat = float(lat_str)
            user_lon = float(lon_str)
            effective_location = f"{user_lat},{user_lon}"
            if location_input:
                effective_location = f"{location_input} ({user_lat}, {user_lon})"
        except ValueError:
            pass

    emergency_type = form_data["emergency_type"]

    # ── Run workflow ──────────────────────────────────────────────────────────
    st.session_state["workflow_result"] = None
    progress_placeholder = st.empty()

    with st.spinner("🔍 Assessing emergency situation..."):
        progress_placeholder.info("🤖 Step 1/3 — Emergency Assessment Agent is analyzing the situation...")

        if user_lat is None and location_input and maps_service.is_configured():
            lat_r, lon_r, _ = maps_service.geocode_location(location_input)
            if lat_r is not None:
                user_lat, user_lon = lat_r, lon_r

        progress_placeholder.info("🗺️ Step 2/3 — Searching for nearby emergency services...")
        progress_placeholder.info("🛡️ Step 3/3 — Preparing emergency safety guidance...")

        workflow_result = agents.run_emergency_workflow(
            emergency_type=emergency_type,
            description=clean_description,       # PII-redacted version
            location=effective_location,
            image_bytes=image_bytes,
            image_mime=image_mime,
        )

    progress_placeholder.empty()

    # Geocoded coordinates from maps result
    if user_lat is None and workflow_result.get("maps_result"):
        search_loc = workflow_result["maps_result"].get("search_location", {})
        user_lat = search_loc.get("latitude")
        user_lon = search_loc.get("longitude")

    workflow_result["user_lat"] = user_lat
    workflow_result["user_lon"] = user_lon
    workflow_result["emergency_type"] = emergency_type
    workflow_result["location_input"] = location_input or effective_location

    # ── Update metrics (Fix 1+2 — structured tokens + cost) ──────────────────
    st.session_state["metrics"] = update_metrics_from_workflow(
        st.session_state["metrics"],
        workflow_result,
        emergency_type=emergency_type,
    )

    # ── Persist to SQLite cross-session memory (Fix 6) ────────────────────────
    _persist_workflow(workflow_result, st.session_state["metrics"])

    st.session_state["workflow_result"] = workflow_result
    _display_results(workflow_result)


def _persist_workflow(workflow_result: dict, metrics: SessionMetrics) -> None:
    """Save a non-sensitive workflow summary to SQLite."""
    try:
        assessment = workflow_result.get("assessment")
        history = metrics.latency_history
        last = history[-1] if history else None
        memory.save_workflow(
            emergency_type=workflow_result.get("emergency_type", "Unknown"),
            severity_level=assessment.severity_level if assessment else "Unknown",
            status=workflow_result.get("status", "failed"),
            total_time_s=workflow_result.get("total_time_s", 0.0),
            assessment_s=last.assessment_s if last else 0.0,
            response_s=last.response_s if last else 0.0,
            maps_s=last.maps_s if last else 0.0,
            input_tokens=last.input_tokens if last else 0,
            output_tokens=last.output_tokens if last else 0,
            cost_usd=last.cost_usd if last else 0.0,
            guardrail_hits=metrics.guardrail_interventions,
            maps_results=len(workflow_result.get("help_centers", [])),
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Memory persist failed: %s", exc)


def _display_results(workflow_result: dict) -> None:
    """Render the full results section from a completed workflow result dict."""
    assessment = workflow_result.get("assessment")
    response = workflow_result.get("response")
    help_centers = workflow_result.get("help_centers", [])
    user_lat = workflow_result.get("user_lat")
    user_lon = workflow_result.get("user_lon")
    emergency_type = workflow_result.get("emergency_type", "")
    location_input = workflow_result.get("location_input", "")
    status = workflow_result.get("status")
    error_message = workflow_result.get("error_message")

    st.markdown("---")

    # Always show the emergency banner in results context
    render_emergency_banner()

    # ── Workflow error ────────────────────────────────────────────────────────
    if status != "success" or assessment is None:
        st.error(
            f"⚠️ The emergency assistance workflow encountered an issue:\n\n"
            f"{error_message or 'Unknown error.'}\n\n"
            f"**Please call 112 for immediate emergency assistance.**"
        )
        if assessment is not None:
            st.markdown("---")
            st.markdown("ℹ️ Partial assessment results are available below:")
            render_assessment(assessment)
        return

    # ── Assessment ────────────────────────────────────────────────────────────
    render_assessment(assessment)

    st.markdown("---")

    # ── Safety guidance ───────────────────────────────────────────────────────
    if response:
        render_response(response)

    st.markdown("---")

    # ── Map + Help centers ────────────────────────────────────────────────────
    st.markdown(f"### 🗺️ Nearby Emergency Services — {category_icon(emergency_type)} {emergency_type}")

    if user_lat and user_lon:
        render_map(help_centers, user_lat, user_lon)

    render_help_centers(help_centers, user_lat, user_lon)

    # ── Agent detail expander ─────────────────────────────────────────────────
    if st.session_state.get("show_agent_details"):
        with st.expander("🔬 Agent Execution Details", expanded=False):
            st.markdown("**Execution Log**")
            for record in workflow_result.get("execution_log", []):
                status_icon = "✅" if record.status == "Completed" else "❌"
                st.markdown(
                    f"{status_icon} **{record.component}** — "
                    f"Status: `{record.status}` | "
                    f"Time: `{record.execution_time_s}s` | "
                    f"Validation: `{record.validation}`"
                )
                if record.notes:
                    st.caption(record.notes)

    # ── Final report + download ───────────────────────────────────────────────
    if assessment and response:
        st.markdown("---")
        st.markdown("### 📄 Emergency Assistance Report")

        report_text = build_emergency_report_text(
            assessment=assessment,
            response=response,
            help_centers=help_centers,
            location_input=location_input,
            emergency_type=emergency_type,
        )

        with st.expander("Preview Report", expanded=False):
            st.text(report_text)

        st.download_button(
            label="⬇️ Download Emergency Assistance Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name="emergency_assistance_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    main()
