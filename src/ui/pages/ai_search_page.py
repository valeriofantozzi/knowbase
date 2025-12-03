import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.ai_search.graph import build_graph
from src.ui.theme import ICONS

def render_ai_search_page():
    """
    Render the AI Search page with chat interface.
    """
    st.title(f"{ICONS.get('search', '🤖')} AI Search Assistant")
    st.markdown("Ask questions about your documents and get AI-generated answers based on the content.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize graph
    if "ai_graph" not in st.session_state:
        with st.spinner("Initializing AI Agent..."):
            try:
                st.session_state.ai_graph = build_graph()
            except Exception as e:
                st.error(f"Failed to initialize AI Agent: {e}")
                return

    # Display chat messages
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to history
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Prepare state
                    initial_state = {
                        "messages": st.session_state.messages,
                        "question": prompt
                    }
                    
                    # Run graph
                    # We use invoke for simplicity
                    response = st.session_state.ai_graph.invoke(initial_state)
                    
                    answer = response.get("generation", "I couldn't generate an answer.")
                    
                    # Append sources if available
                    documents = response.get("documents", [])
                    if documents:
                        answer += "\n\n---\n**Sources:**\n"
                        for i, doc in enumerate(documents, 1):
                            source_name = doc.metadata.filename if hasattr(doc.metadata, 'filename') else "Unknown"
                            score = f"{doc.similarity_score:.2f}" if doc.similarity_score else "N/A"
                            # Create a snippet (first 100 chars)
                            snippet = doc.text[:150].replace("\n", " ") + "..."
                            answer += f"{i}. `{source_name}` (Score: {score})\n   > {snippet}\n"

                    message_placeholder.markdown(answer)
                    
                    # Add AI message to history
                    st.session_state.messages.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
