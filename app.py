import base64
import json
import pandas as pd
import streamlit as st

from src import utilis


# function to display a PDF file
def show_pdf(file_path):
    # Using an iframe to display the PDF
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1000" height="1000" type="application/pdf"></iframe>'
        return pdf_display


if __name__ == "__main__":
    st.set_page_config(page_title="Talent Scout", page_icon="🔍", layout="wide")

    emoji = "🔍"
    st.write(
        f'<span style="font-size: 75px; line-height: 1">Talent Scout {emoji}</span>',
        unsafe_allow_html=True,
    )

    """
    ### Talent Scout

    [![](https://img.shields.io/badge/thivav-github-repo)](https://github.com/ThivaV) &nbsp;
    [![](https://img.shields.io/github/stars/thivav/ats-tracker?style=social)](https://github.com/ThivaV/ats-tracker)
    """

    # initialize the database and retriever operations
    resume_catalog_uri = r"data/processed_data/resumes.csv"
    milvusdb_uri = r"db/milvus/ATSTracker_BM25.db"
    collection_name = "ats_tracker_resumes_collection"
    selected_resume_path = None

    if "handler" not in st.session_state:
        st.session_state.handler = utilis.ATSTracker(
            resume_catalog_uri, milvusdb_uri, collection_name
        )
        st.session_state.handler.initialize_catalog()
        st.session_state.handler.initialize_database()
        st.session_state.handler.initialize_sparse_embeddings()
        st.session_state.handler.initialize_dense_embeddings()

    col1, col2 = st.columns(2)
    with col1:
        sparse_weight = st.slider("Sparse Embedding Weight", 0.0, 1.0, 0.5, 0.1)
        dense_weight = st.slider("Dense Embedding Weight", 0.0, 1.0, 0.5, 0.1)
        top_k = st.number_input(
            "Enter the number of resumes to display",
            min_value=5,
            max_value=200,
            value=30,
        )

        jd = st.text_area(
            "Enter job description 👇", placeholder="Job Description", height=300
        )

        if jd.strip():
            resumes = pd.DataFrame(
                json.loads(
                    st.session_state.handler.search(
                        jd, sparse_weight, dense_weight, top_k
                    )
                )
            )

            event = st.dataframe(
                resumes,
                column_config={
                    "resume_id": st.column_config.Column(
                        "Resume ID", help="Resume ID", width="medium", disabled=True
                    ),
                    "domain": st.column_config.Column(
                        "Resume", help="Resume category", width="medium", disabled=True
                    ),
                    "uri": st.column_config.LinkColumn(
                        "Resume",
                        display_text="Open Resume",
                        help="Resume uri",
                        width="medium",
                        disabled=True,
                    ),
                    "distance": st.column_config.Column(
                        "Match (%)",
                        help="Resume matching percentage with the job description",
                        width="medium",
                        disabled=True,
                    ),
                },
                column_order=["resume_id", "domain", "uri", "distance"],
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["single-row"],
            )

            if len(event.selection.rows) > 0:
                df_selected = resumes.loc[event.selection.rows]
                selected_resume_path = df_selected["uri"].values[0].lstrip("../")
    with col2:
        if selected_resume_path:
            pdf_display = show_pdf(selected_resume_path)
            st.markdown(pdf_display, unsafe_allow_html=True)
