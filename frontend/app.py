"""Streamlit MVP frontend for the Synapse Council with real-time progress."""

import time
import streamlit as st
import httpx
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 1  # seconds - fast polling for real-time updates

# Page configuration
st.set_page_config(
    page_title="Synapse Council",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .phase-indicator {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    .phase-active {
        background: #667eea;
        color: white;
    }
    .phase-complete {
        background: #28a745;
        color: white;
    }
    .phase-pending {
        background: #e9ecef;
        color: #6c757d;
    }
    .interrupt-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .mermaid-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .proposal-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "status" not in st.session_state:
        st.session_state.status = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "question" not in st.session_state:
        st.session_state.question = None


def make_request(method: str, endpoint: str, data: Optional[dict] = None, timeout: int = 120) -> dict:
    """Make an HTTP request to the API."""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "GET":
                response = client.get(url)
            elif method == "POST":
                response = client.post(url, json=data)
            elif method == "DELETE":
                response = client.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as e:
        st.error(f"API Error: {e}")
        return {"error": str(e)}


def render_sidebar():
    """Render the sidebar with system context form."""
    st.sidebar.markdown("## System Context (Optional)")

    with st.sidebar.expander("📋 Configure Context", expanded=False):
        company = st.text_input("Company/Project Name")
        domain = st.selectbox(
            "Business Domain",
            ["", "fintech", "healthcare", "e-commerce", "social", "enterprise", "other"],
        )
        tech_stack = st.text_area(
            "Current Tech Stack (comma-separated)",
            placeholder="Python, PostgreSQL, Redis, Kubernetes",
        )

        st.markdown("### Constraints")
        budget = st.text_input("Budget Constraint", placeholder="e.g., $100k/month")
        team_size = st.number_input("Team Size", min_value=1, value=5)
        timeline = st.text_input("Timeline", placeholder="e.g., 6 months")

        st.markdown("### Performance SLAs")
        latency = st.text_input("Latency Target", placeholder="e.g., <100ms p99")
        uptime = st.text_input("Uptime Target", placeholder="e.g., 99.99%")

    # Build context object
    context = None
    if any([company, domain, tech_stack, budget, latency]):
        context = {
            "company_name": company or None,
            "domain": domain or None,
            "current_tech_stack": [t.strip() for t in tech_stack.split(",")] if tech_stack else [],
            "constraints": [],
            "performance_slas": [],
        }

        if budget:
            context["constraints"].append({
                "type": "budget",
                "description": budget,
                "severity": "hard",
            })

        if timeline:
            context["constraints"].append({
                "type": "timeline",
                "description": timeline,
                "severity": "hard",
            })

        if team_size:
            context["team"] = {
                "size": team_size,
                "expertise": [],
                "experience_level": "senior",
            }

        if latency:
            context["performance_slas"].append({
                "metric": "latency",
                "target": latency,
                "priority": "high",
            })

        if uptime:
            context["performance_slas"].append({
                "metric": "uptime",
                "target": uptime,
                "priority": "critical",
            })

    return context


def render_phase_timeline(current_phase: str):
    """Render a visual phase timeline."""
    phases = [
        ("Ideation", ["start", "ideation"]),
        ("Critique 1", ["ideation_complete", "cross_critique"]),
        ("Refinement", ["cross_critique_complete", "refinement"]),
        ("Critique 2", ["refinement_complete", "cross_critique_2"]),
        ("Audit", ["cross_critique_2_complete", "audit"]),
        ("Complete", ["audit_complete", "convergence", "complete", "escalated"]),
    ]
    
    phase_order = [
        "start", "ideation", "ideation_complete", 
        "cross_critique", "cross_critique_complete",
        "refinement", "refinement_complete",
        "cross_critique_2", "cross_critique_2_complete",
        "audit", "audit_complete", 
        "convergence", "complete", "escalated"
    ]
    
    current_idx = phase_order.index(current_phase) if current_phase in phase_order else 0
    
    cols = st.columns(len(phases))
    for i, (phase_name, phase_values) in enumerate(phases):
        phase_idx = phase_order.index(phase_values[0])
        
        if current_phase in phase_values:
            status_class = "phase-active"
            icon = "🔄"
        elif current_idx > phase_idx:
            status_class = "phase-complete"
            icon = "✅"
        else:
            status_class = "phase-pending"
            icon = "⏳"
        
        with cols[i]:
            st.markdown(
                f'<div class="phase-indicator {status_class}">{icon} {phase_name}</div>',
                unsafe_allow_html=True
            )


def render_agent_proposal(title: str, proposal: dict, icon: str, color: str):
    """Render an agent's proposal card."""
    if not proposal:
        st.info(f"{icon} {title} is working on their proposal...")
        return
    
    with st.expander(f"{icon} **{title}**: {proposal.get('title', 'Proposal')}", expanded=True):
        # Confidence bar
        confidence = proposal.get('confidence', 0)
        st.progress(confidence, text=f"Confidence: {confidence:.0%}")
        
        # Summary
        st.markdown(f"**Summary:** {proposal.get('summary', 'N/A')}")
        
        # Full Approach (always show complete)
        approach = proposal.get('approach', '')
        if approach:
            st.markdown(f"**Approach:**")
            st.markdown(approach)
        
        # Components (always expanded)
        components = proposal.get('components', [])
        if components:
            st.markdown(f"**🧩 Components ({len(components)}):**")
            for c in components:
                if isinstance(c, dict):
                    st.markdown(f"- **{c.get('name', 'N/A')}** ({c.get('type', 'N/A')}): {c.get('technology', 'N/A')} - {c.get('description', 'N/A')}")
                else:
                    st.markdown(f"- {c}")
        
        # Trade-offs (always expanded)
        trade_offs = proposal.get('trade_offs', [])
        if trade_offs:
            st.markdown(f"**⚖️ Trade-offs ({len(trade_offs)}):**")
            for t in trade_offs:
                if isinstance(t, dict):
                    st.markdown(f"- **{t.get('aspect', 'N/A')}**: {t.get('choice', 'N/A')} ({t.get('rationale', 'N/A')})")
                else:
                    st.markdown(f"- {t}")
        
        # Risks (if available)
        risks = proposal.get('risks', [])
        if risks:
            st.markdown(f"**⚠️ Risks ({len(risks)}):**")
            for r in risks:
                if isinstance(r, dict):
                    st.markdown(f"- [{r.get('severity', 'N/A').upper()}] {r.get('category', 'N/A')}: {r.get('description', 'N/A')} - Mitigation: {r.get('mitigation', 'N/A')}")
                else:
                    st.markdown(f"- {r}")
        
        # Diagram (if available)
        diagram = proposal.get('mermaid_diagram', '')
        if diagram:
            st.markdown("**📊 Architecture Diagram:**")
            st.code(diagram, language="mermaid")


def render_progress(status: dict):
    """Render the progress view with agent responses."""
    current_phase = status.get("current_phase", "start")
    is_running = status.get("is_running", False)
    elapsed = status.get("elapsed_time")
    
    # Phase timeline
    st.markdown("### 📊 Progress")
    render_phase_timeline(current_phase)
    
    # Status info
    col1, col2, col3 = st.columns(3)
    with col1:
        phase_display = current_phase.replace("_", " ").title() if current_phase else "Starting..."
        st.metric("Current Phase", phase_display)
    with col2:
        if elapsed:
            st.metric("Elapsed Time", f"{elapsed:.0f}s")
    with col3:
        if is_running:
            st.markdown("🔄 **Status:** Running")
        elif status.get("is_complete"):
            st.markdown("✅ **Status:** Complete")
        elif status.get("is_waiting_for_input"):
            st.markdown("⏸️ **Status:** Waiting for input")
        else:
            st.markdown("⏹️ **Status:** Stopped")
    
    st.markdown("---")
    
    # Agent Proposals
    st.markdown("### 🤖 Agent Proposals")
    col1, col2 = st.columns(2)
    
    with col1:
        render_agent_proposal(
            "The Architect", 
            status.get("architect_proposal"),
            "🏗️",
            "#667eea"
        )
    
    with col2:
        render_agent_proposal(
            "The Engineer",
            status.get("engineer_proposal"),
            "⚙️",
            "#764ba2"
        )
    
    # Critiques Phase 1 - show full content
    if status.get("architect_critique_summary") or status.get("engineer_critique_summary"):
        st.markdown("### 🔄 Cross-Critique (Round 1)")
        col1, col2 = st.columns(2)
        with col1:
            if status.get("architect_critique_summary"):
                critique_text = status['architect_critique_summary']
                with st.expander("🏗️ **Architect's critique of Engineer**", expanded=True):
                    st.markdown(critique_text)
            else:
                st.info("🏗️ Architect's critique pending...")
        with col2:
            if status.get("engineer_critique_summary"):
                critique_text = status['engineer_critique_summary']
                with st.expander("⚙️ **Engineer's critique of Architect**", expanded=True):
                    st.markdown(critique_text)
            else:
                st.info("⚙️ Engineer's critique pending...")
    
    # Refined Proposals
    if status.get("architect_refined_proposal") or status.get("engineer_refined_proposal"):
        st.markdown("### ✨ Refined Proposals")
        col1, col2 = st.columns(2)
        
        with col1:
            render_agent_proposal(
                "The Architect (Refined)", 
                status.get("architect_refined_proposal"),
                "🏗️",
                "#667eea"
            )
        
        with col2:
            render_agent_proposal(
                "The Engineer (Refined)",
                status.get("engineer_refined_proposal"),
                "⚙️",
                "#764ba2"
            )
    
    # Critiques Phase 2
    if status.get("architect_critique_2_summary") or status.get("engineer_critique_2_summary"):
        st.markdown("### 🔄 Cross-Critique (Round 2)")
        col1, col2 = st.columns(2)
        with col1:
            if status.get("architect_critique_2_summary"):
                critique_text = status['architect_critique_2_summary']
                with st.expander("🏗️ **Architect's critique of refined Engineer**", expanded=True):
                    st.markdown(critique_text)
            else:
                st.info("🏗️ Architect's second critique pending...")
        with col2:
            if status.get("engineer_critique_2_summary"):
                critique_text = status['engineer_critique_2_summary']
                with st.expander("⚙️ **Engineer's critique of refined Architect**", expanded=True):
                    st.markdown(critique_text)
            else:
                st.info("⚙️ Engineer's second critique pending...")
    
    # Audit
    if status.get("audit_preferred"):
        st.markdown("### 🔍 Audit Result")
        preferred = status.get("audit_preferred", "").title()
        consensus = "Yes" if status.get("audit_consensus_possible") else "No"
        st.success(f"**Preferred approach:** {preferred} | **Consensus possible:** {consensus}")
    
    # Usage metrics
    if status.get("usage_metrics"):
        with st.expander("📊 Usage Metrics"):
            metrics = status["usage_metrics"]
            st.json(metrics)


def render_interrupt(status: dict):
    """Render an interrupt requiring user input."""
    if not status.get("is_waiting_for_input"):
        return
    
    st.markdown("---")
    st.markdown('<div class="interrupt-box">', unsafe_allow_html=True)

    source = status.get("interrupt_source") or "agent"
    question = status.get("interrupt_question") or "Please provide more information."

    st.markdown(f"### 🤔 The {source.title()} needs clarification:")
    st.markdown(f"**{question}**")

    response = st.text_area(
        "Your response:",
        key="clarification_response",
        placeholder="Provide your answer here...",
    )

    if st.button("Submit Response", type="primary"):
        if response:
            with st.spinner("Processing your response..."):
                result = make_request(
                    "POST",
                    f"/api/design/{st.session_state.session_id}/respond",
                    {"response": response},
                )
                st.session_state.status = result
                st.session_state.auto_refresh = True
                st.rerun()
        else:
            st.error("Please provide a response.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_result(result: dict):
    """Render the final result."""
    st.markdown("---")
    st.markdown('<div class="success-box">', unsafe_allow_html=True)

    if result.get("escalated"):
        st.warning("⚠️ The council could not reach consensus.")
        st.markdown(result.get("error", ""))
    else:
        adr = result.get("adr", {})
        st.markdown(f"## ✅ {adr.get('title', 'Architecture Decision')}")

        st.markdown(f"**Status:** {adr.get('status', 'proposed')}")
        st.markdown(f"**Consensus Level:** {adr.get('consensus_level', 0):.0%}")
        st.markdown(f"**Rounds Taken:** {adr.get('rounds_taken', 1)}")

        # Decision
        st.markdown("### 📋 Decision")
        st.markdown(adr.get("decision", ""))

        # Rationale
        st.markdown("### 💡 Rationale")
        st.markdown(adr.get("rationale", ""))

        # Final Proposal - Combined best approach with full details
        st.markdown("### 🎯 Final Proposal")
        
        # Get majority opinion for detailed display
        majority = adr.get("majority_opinion", {})
        
        # Show the full approach/decision
        final_approach = adr.get("final_approach") or majority.get("approach") or adr.get("decision", "")
        if final_approach:
            st.markdown("**Recommended Approach:**")
            st.markdown(final_approach)
        
        # Show complete components from majority
        if majority.get("components"):
            st.markdown(f"**🧩 Components ({len(majority['components'])}):**")
            for c in majority["components"]:
                st.markdown(f"- **{c.get('name', 'N/A')}** ({c.get('type', 'N/A')}): {c.get('technology', 'N/A')} - {c.get('description', 'N/A')}")
        
        # Show trade-offs from majority
        if majority.get("trade_offs"):
            st.markdown(f"**⚖️ Trade-offs ({len(majority['trade_offs'])}):**")
            for t in majority["trade_offs"]:
                st.markdown(f"- **{t.get('aspect', 'N/A')}**: {t.get('choice', 'N/A')} ({t.get('rationale', 'N/A')})")
        
        # Implementation recommendations
        impl_recommendations = adr.get("implementation_recommendations", [])
        if impl_recommendations:
            st.markdown(f"**📝 Implementation Recommendations ({len(impl_recommendations)}):**")
            for rec in impl_recommendations:
                st.markdown(f"- {rec}")
        
        # Diagram in Final Proposal
        diagram = majority.get("mermaid_diagram") or adr.get("mermaid_diagram")
        if diagram:
            st.markdown("**📊 Architecture Diagram:**")
            st.code(diagram, language="mermaid")
        
        # Risks in Final Proposal
        majority_risks = majority.get("risks", [])
        adr_risks = adr.get("risks", [])
        all_risks = majority_risks if majority_risks else adr_risks
        if all_risks:
            st.markdown(f"**⚠️ Identified Risks ({len(all_risks)}):**")
            for risk in all_risks:
                if isinstance(risk, dict):
                    st.markdown(f"- [{risk.get('severity', 'N/A').upper()}] {risk.get('category', 'N/A')}: {risk.get('description', 'N/A')} - Mitigation: {risk.get('mitigation', 'N/A')}")
                else:
                    st.markdown(f"- {risk}")

        # Minority Report
        minority = adr.get("minority_report")
        if minority:
            st.markdown("### 📝 Minority Report")
            with st.expander("View dissenting opinion", expanded=False):
                st.markdown(minority)

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application."""
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">🧠 The Synapse Council</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">An AI-powered Architecture Review Board for System Design</p>',
        unsafe_allow_html=True,
    )

    # Model options for each agent
    OPENAI_MODELS = ["o3", "o3-mini", "o1", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    ANTHROPIC_MODELS = ["claude-opus-4-20250514", "claude-sonnet-4-20250514"]  # Opus 4 for best reasoning
    GOOGLE_MODELS = ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"]

    # Initialize model selections in session state
    if "architect_model" not in st.session_state:
        st.session_state.architect_model = "gpt-4o"
    if "engineer_model" not in st.session_state:
        st.session_state.engineer_model = "claude-sonnet-4-20250514"
    if "auditor_model" not in st.session_state:
        st.session_state.auditor_model = "gemini-2.5-pro"

    # Component Descriptions with Model Selection
    with st.expander("ℹ️ Configure the Council", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🏗️ The Architect")
            st.markdown("Focuses on high-level system design, patterns, and trade-offs.")
            st.session_state.architect_model = st.selectbox(
                "Architect Model",
                options=OPENAI_MODELS,
                index=OPENAI_MODELS.index(st.session_state.architect_model) if st.session_state.architect_model in OPENAI_MODELS else 0,
                key="architect_model_select"
            )
        with col2:
            st.markdown("### ⚙️ The Engineer")
            st.markdown("Focuses on implementation details, technologies, and reliability.")
            st.session_state.engineer_model = st.selectbox(
                "Engineer Model",
                options=ANTHROPIC_MODELS,
                index=ANTHROPIC_MODELS.index(st.session_state.engineer_model) if st.session_state.engineer_model in ANTHROPIC_MODELS else 0,
                key="engineer_model_select"
            )
        with col3:
            st.markdown("### 🔍 The Auditor")
            st.markdown("Evaluates both proposals for security, scalability, and consensus.")
            st.session_state.auditor_model = st.selectbox(
                "Auditor Model",
                options=GOOGLE_MODELS,
                index=GOOGLE_MODELS.index(st.session_state.auditor_model) if st.session_state.auditor_model in GOOGLE_MODELS else 0,
                key="auditor_model_select"
            )

    # Sidebar
    context = render_sidebar()

    # Main content
    st.markdown("---")

    # Input form - only show if no active session
    if not st.session_state.session_id:
        st.markdown("### 💬 Ask your system design question")

        with st.form("design_question_form"):
            question = st.text_area(
                "Describe the system you want to design:",
                height=150,
                placeholder="Example: Design a real-time collaborative document editing system like Google Docs that needs to handle 10 million concurrent users with sub-100ms latency for global users.",
            )

            submit = st.form_submit_button("🚀 Start Design Review", type="primary")

        # Handle form submission OUTSIDE the form context for proper st.rerun() behavior
        if submit:
            if not question:
                st.error("Please enter a system design question.")
            else:
                with st.spinner("Starting design review..."):
                    # Use the async start endpoint
                    request_data = {
                        "question": question,
                        "architect_model": st.session_state.architect_model,
                        "engineer_model": st.session_state.engineer_model,
                        "auditor_model": st.session_state.auditor_model,
                    }
                    if context:
                        request_data["system_context"] = context

                    result = make_request("POST", "/api/design/start", request_data)

                    if not result.get("error"):
                        st.session_state.session_id = result.get("session_id")
                        st.session_state.status = result
                        st.session_state.question = question  # Persist question
                        st.session_state.auto_refresh = True
                        st.rerun()
                    else:
                        st.error(f"Failed to start design review: {result.get('error')}")

    else:
        # Active session - show progress
        
        # Display the persistent question
        if st.session_state.question:
            st.info(f"🎯 **Design Goal:** {st.session_state.question}")
        
        status = st.session_state.status
        
        if status:
            # Render progress view
            render_progress(status)
            
            # Handle interrupt
            if status.get("is_waiting_for_input"):
                render_interrupt(status)
                st.session_state.auto_refresh = False
            
            # Show result if complete
            elif status.get("is_complete"):
                st.session_state.auto_refresh = False
                
                # Fetch full result
                if not st.session_state.result:
                    result = make_request(
                        "GET",
                        f"/api/design/{st.session_state.session_id}/result",
                    )
                    st.session_state.result = result

                render_result(st.session_state.result)

                # Reset button
                st.markdown("---")
                if st.button("🔄 Start New Design"):
                    st.session_state.session_id = None
                    st.session_state.status = None
                    st.session_state.result = None
                    st.session_state.auto_refresh = False
                    st.rerun()
            
            else:
                # Still running - show refresh controls
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔄 Refresh Now"):
                        status = make_request(
                            "GET",
                            f"/api/design/{st.session_state.session_id}/detailed",
                        )
                        st.session_state.status = status
                        st.rerun()
                
                with col2:
                    auto = st.checkbox("Auto-refresh every second", value=st.session_state.auto_refresh)
                    st.session_state.auto_refresh = auto
                
                # Auto-refresh logic
                if st.session_state.auto_refresh:
                    # Update status before sleeping/rerunning
                    status = make_request("GET", f"/api/design/{st.session_state.session_id}/detailed")
                    st.session_state.status = status
                    
                    # Render console before sleep/rerun so it's visible during the wait
                    render_system_console()
                    
                    time.sleep(POLL_INTERVAL)
                    st.rerun()

        # Stop session button
        st.markdown("---")
        col_cancel, col_console = st.columns([1, 2])
        with col_cancel:
            if st.button("⏹️ Cancel and Start Over"):
                st.session_state.session_id = None
                st.session_state.status = None
                st.session_state.result = None
                st.session_state.auto_refresh = False
                st.rerun()
    
    # Render console if not already rendered by auto-refresh logic
    # (i.e., when stopped, complete, or waiting for input)
    if not st.session_state.auto_refresh:
        render_system_console()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Powered by <strong>GPT-4o</strong> (Architect) •
            <strong>Claude Opus 4</strong> (Engineer) •
            <strong>Gemini 2.5 Pro</strong> (Auditor)
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_console():
    """Render the system console with backend logs."""
    with st.expander("🖥️ System Console (Backend Logs)", expanded=True):
        # Add a manual refresh button for logs inside the expander
        if st.button("Refresh Logs", key="refresh_logs"):
            pass  # Clicking this will rerun the script and fetch new logs
            
        logs_data = make_request("GET", "/api/logs?lines=500")
        if logs_data and "logs" in logs_data:
            # Join logs and display in a scrollable div that auto-scrolls to bottom
            log_text = "".join(logs_data["logs"])
            # Escape HTML special characters
            import html
            escaped_logs = html.escape(log_text)
            
            # Custom HTML with auto-scroll to bottom
            console_html = f"""
            <div id="log-console" style="
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 10px;
                border-radius: 5px;
                height: 400px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            ">{escaped_logs}</div>
            <script>
                var logConsole = document.getElementById('log-console');
                logConsole.scrollTop = logConsole.scrollHeight;
            </script>
            """
            st.components.v1.html(console_html, height=420, scrolling=False)
        else:
            st.info("No logs available.")


if __name__ == "__main__":
    main()
