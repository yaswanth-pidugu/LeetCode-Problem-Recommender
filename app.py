import streamlit as st
from app.recommender import load_resources, get_recommendations

st.set_page_config(
    page_title="LeetCode Problem Recommender",
    layout="wide"
)

st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #374151;
        margin-bottom: 25px;
    }
    .highlight {
        background-color: #E5E7EB;
        padding: 8px 15px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>LeetCode Problem Recommender System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Find similar LeetCode problems based on content, tags, difficulty, and popularity.</div>",
    unsafe_allow_html=True
)

@st.cache_resource
def load_all():
    return load_resources(
        data_path="Data_Pipeline/preprocessed_data.csv",
        model_path="models/lightgbm_model.pkl",
        embed_cache="sbert_embeddings.pkl"
    )

with st.spinner("Loading model and data... Please wait."):
    df, emb, tag, diff, pop, model = load_all()


st.markdown("### Input Section")
st.write("Enter the LeetCode **problem number** (as seen on the LeetCode website).")

problem_number = st.number_input(
    "Problem Number",
    min_value=1,
    max_value=len(df),
    step=1,
    value=1,
    help="Enter the exact LeetCode problem number (e.g., 1 for Two Sum)"
)

if st.button("Generate Recommendations"):
    with st.spinner("Generating recommendations..."):
        index = problem_number - 1  # Adjust for zero-based indexing
        base_title = df.iloc[index]['title']

        st.markdown(
            f"<div class='highlight'><strong>Base Problem:</strong> {base_title}</div>",
            unsafe_allow_html=True
        )

        # Get recommendations
        recs = get_recommendations(index, df, emb, tag, diff, pop, model, k=10)


        st.markdown("### Recommended Problems")

        # Create LeetCode URLs based on title pattern
        import re
        def make_url(title):
            # Remove leading numbers, dots, and extra spaces
            clean_title = re.sub(r'^\d+[\.\s-]*', '', title.strip())
            # Replace remaining spaces or underscores with hyphens
            clean_title = re.sub(r'[\s_]+', '-', clean_title.lower())
            # Remove any trailing punctuation
            clean_title = re.sub(r'[^a-z0-9\-]', '', clean_title)
            return f"https://leetcode.com/problems/{clean_title}/"


        for _, row in recs.iterrows():
            title = row['title']
            url = make_url(title)
            tags = ", ".join(row['topic_tags']) if isinstance(row['topic_tags'], list) else row['topic_tags']
            difficulty = row['difficulty']

            st.markdown(
                f"""
                <div style='background-color:#F3F4F6;padding:15px;border-radius:10px;margin-bottom:10px;'>
                    <a href='{url}' target='_blank' style='font-size:18px;font-weight:600;color:#2563EB;text-decoration:none;'>{title}</a>
                    <div style='color:#374151;font-size:15px;margin-top:3px;'>
                        <strong>Difficulty:</strong> {difficulty} | <strong>Tags:</strong> {tags}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    st.info("Enter a valid problem number above and click **Generate Recommendations**.")