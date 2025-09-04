def run_chatbot_agent():
    import streamlit as st
    st.subheader("🤖 AI Financial Chatbot")

    st.info("Ask questions about your analyzed financial documents")

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your financial AI assistant. Ask me anything about the analyzed documents."}
        ]

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the financial analysis..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            response = f"Thank you for asking about: '{prompt}'. Based on the analyzed financial documents, I can provide insights about financial performance, risk analysis, and market trends. This is a demo response - in a full implementation, I would analyze your specific document content."
            st.markdown(response)

        # Add assistant message
        st.session_state.chat_messages.append({"role": "assistant", "content": response})