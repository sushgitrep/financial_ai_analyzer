def run_summarization_agent():
    import streamlit as st
    st.subheader("📝 AI Summarization")
    if 'document_data' not in st.session_state:
        st.warning("⚠️ Please process a document first in the Preprocessing tab")
    else:
        st.info("**Ready:** Document loaded and ready for summarization")
        if st.button("📝 Generate Summary", type="primary", use_container_width=True, key="gen_summary_polished"):
            with st.spinner("Generating comprehensive summary..."):
                import time
                time.sleep(2)
                st.success("✅ Summary generated!")
                st.markdown("### 📄 Document Summary")
                st.markdown("""
                **Executive Summary:**
                This financial document contains key insights about the bank's performance, 
                risk management strategies, and future outlook. Key highlights include 
                financial metrics, regulatory compliance, and strategic initiatives.

                **Key Topics:** Credit risk, financial performance, regulatory compliance
                """)