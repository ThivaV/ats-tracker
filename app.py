import base64
import streamlit as st

from src import utilis


# Function to display a PDF file
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
    milvusdb_uri = r"db/milvus/ats_tracker.db"
    collection_name = "ats_tracker_resumes_collection"

    handler = utilis.HandleATS(resume_catalog_uri, milvusdb_uri, collection_name)
    handler.initialize_catalog()
    handler.initialize_embedding()
    handler.initialize_database()

    col1, col2 = st.columns(2)
    with col1:
        jd = st.text_input("Enter job description 👇", placeholder="Job Description")
        resume_ids = handler.search(jd, 0.5, 0.5, 5)
        resumes = handler.retrieve_resume_info(resume_ids)

        ui_resumes = resumes.drop(columns=["resume"])
        event = st.dataframe(
            ui_resumes,
            column_config={
                "resume_id": st.column_config.Column("Resume ID", width="small"),
                "resume_domain": st.column_config.Column("Domain", width="large"),
                "resume_uri": st.column_config.LinkColumn(
                    "Resume", display_text="Open Resume", width="large"
                ),
            },
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["single-row"],
        )
        selection = event.selection        
        selection_index = int(selection.rows[0])    

        # st.write(ui_resumes[selection_index, "resume_uri"])
    with col2:
        pdf_display = show_pdf("data/master_data/resumes/v1.0/AGRICULTURE/62994611.pdf")
        st.markdown(pdf_display, unsafe_allow_html=True)
