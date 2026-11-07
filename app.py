"""Simple web page for the agent. Run with: streamlit run app.py"""
import streamlit as st

from agent import MODEL, ask

st.set_page_config(page_title="Text-to-SQL Agent", page_icon="🗄️")
st.title("Text-to-SQL Agent")
st.caption("Ask a question about a music store database. The agent writes and runs the SQL.")
st.caption(f"Model: {MODEL}")

# EXAMPLES = [
#     "Top 5 customers by total spending",
#     "Which genre has sold the most tracks?",
#     "How many tracks have never been purchased?",
# ]
# st.write("Try: " + " | ".join(f"*{e}*" for e in EXAMPLES))

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("The agent is working..."):
        out = ask(question.strip())

    if out["finished"] and out["answer"]:
        st.subheader("Answer")
        st.write(out["answer"])
    else:
        st.warning("The agent could not finish within its step limit. The log below shows where it got stuck.")

    if out["sql"]:
        st.subheader("SQL that produced the answer")
        st.code(out["sql"], language="sql")

    st.caption(f"Failed queries along the way: {out['errors']}")

    with st.expander("Agent steps"):
        for s in out["steps"]:
            st.markdown(f"**Step {s['step']}: `{s['tool']}`**")
            if s["tool"] == "run_sql":
                st.code(s["args"].get("query", ""), language="sql")
            st.json(s["result"] if isinstance(s["result"], (dict, list)) else {"result": s["result"]})
