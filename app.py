
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Index | Document Discovery",
    page_icon="⌕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# VISUAL DESIGN — THE RESEARCH DESK
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --paper: #f5f3ed;
    --white: #fffefa;
    --ink: #17243b;
    --muted: #778093;
    --blue: #315bea;
    --line: #e4e1d8;
    --soft: #eaeefb;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, p, label {
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    letter-spacing: -1.5px;
}

h2, h3 {
    font-weight: 800 !important;
    letter-spacing: -0.7px;
}

[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace;
    font-size: 10px !important;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    color: var(--blue) !important;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button {
    background: var(--blue);
    color: white;
    border: 1px solid var(--blue);
    border-radius: 4px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #2448c7;
    color: white;
    border-color: #2448c7;
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background: var(--white);
    border-color: var(--line);
    border-radius: 4px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
}

hr {
    border-color: var(--line);
}

[data-testid="stSidebar"] {
    background: #eeece5;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: var(--ink);
}

.brand {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 3px;
    color: var(--blue);
}

.topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--line);
    padding-bottom: 15px;
    margin-bottom: 38px;
}

.top-meta {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}

.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--blue);
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(38px, 6vw, 66px);
    line-height: 1.05;
    letter-spacing: -2px;
    color: var(--ink);
}

.hero-title span {
    color: var(--blue);
    font-style: italic;
}

.hero-description {
    color: var(--muted);
    max-width: 690px;
    font-size: 15px;
    line-height: 1.8;
    margin-top: 18px;
}

.section-label {
    font-family: 'DM Mono', monospace;
    color: var(--blue);
    font-size: 10px;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    color: var(--ink);
    margin-bottom: 4px;
}

.section-description {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 22px;
}

.document-card {
    background: var(--white);
    border: 1px solid var(--line);
    border-left: 3px solid var(--blue);
    padding: 18px;
    margin-bottom: 12px;
}

.document-card small {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}

.document-card strong {
    display: block;
    color: var(--ink);
    font-size: 16px;
    margin: 8px 0;
}

.document-card p {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
}

.search-panel {
    background: var(--white);
    border: 1px solid var(--line);
    padding: 24px;
    margin-bottom: 18px;
}

.topic-panel {
    background: var(--white);
    border: 1px solid var(--line);
    padding: 18px;
    margin-bottom: 12px;
}

.footer {
    border-top: 1px solid var(--line);
    margin-top: 40px;
    padding-top: 18px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# DEMO DOCUMENT COLLECTION
# ==========================================================

@st.cache_data
def create_demo_documents():

    documents = [
        (
            "The Future of Artificial Intelligence",
            "Artificial intelligence is transforming industries through "
            "machine learning deep learning neural networks and automation. "
            "Organizations are adopting AI systems to improve productivity "
            "and automate repetitive tasks."
        ),
        (
            "Machine Learning in Healthcare",
            "Machine learning helps healthcare professionals identify "
            "patterns in medical data. Predictive models support disease "
            "detection patient monitoring medical imaging and clinical "
            "decision support."
        ),
        (
            "Deep Learning and Neural Networks",
            "Deep learning uses multiple layers of neural networks to "
            "learn complex patterns. Computer vision natural language "
            "processing and speech recognition use deep learning methods."
        ),
        (
            "Natural Language Processing",
            "Natural language processing enables computers to understand "
            "human language. Text classification sentiment analysis "
            "information extraction and document search are common NLP tasks."
        ),
        (
            "Cybersecurity Threat Detection",
            "Cybersecurity teams monitor network traffic system logs "
            "and security events to identify malicious activity. Threat "
            "detection uses anomaly analysis intrusion detection and "
            "incident response."
        ),
        (
            "Cloud Computing Infrastructure",
            "Cloud computing provides scalable infrastructure storage "
            "and computing resources. Organizations use cloud platforms "
            "for application hosting data processing and distributed systems."
        ),
        (
            "Data Engineering Pipelines",
            "Data engineering builds reliable pipelines for collecting "
            "transforming and storing information. ETL workflows data "
            "warehouses and data lakes support analytics."
        ),
        (
            "Big Data Analytics",
            "Big data analytics processes large datasets to discover "
            "trends and patterns. Distributed computing data platforms "
            "and visualization tools support data-driven decisions."
        ),
        (
            "Renewable Energy Technologies",
            "Renewable energy includes solar wind hydroelectric and "
            "geothermal power. Energy storage and smart grids help "
            "integrate renewable electricity into power systems."
        ),
        (
            "Climate Change Research",
            "Climate research studies temperature changes greenhouse "
            "gas emissions extreme weather and environmental impacts. "
            "Climate models help researchers understand future scenarios."
        ),
        (
            "Electric Vehicle Technology",
            "Electric vehicles use battery systems electric motors "
            "and power electronics. Battery management charging "
            "infrastructure and vehicle efficiency are important topics."
        ),
        (
            "Smart Grid Management",
            "Smart grids use digital communication sensors and "
            "automation to manage electricity distribution. Grid "
            "optimization supports energy reliability and efficiency."
        ),
        (
            "Digital Learning Platforms",
            "Digital learning platforms provide online courses "
            "interactive lessons assessments and educational content. "
            "Learning analytics can improve student engagement."
        ),
        (
            "Educational Data Mining",
            "Educational data mining studies student performance "
            "learning behaviour and course engagement. Predictive "
            "analytics can identify students who may need support."
        ),
        (
            "Computer Vision Applications",
            "Computer vision enables machines to interpret images "
            "and videos. Object detection image classification "
            "facial recognition and visual inspection are applications."
        ),
        (
            "Robotics and Automation",
            "Robotics combines sensors control systems mechanical "
            "design and artificial intelligence. Industrial robots "
            "automate manufacturing assembly and inspection."
        ),
        (
            "Internet of Things",
            "Internet of Things devices use sensors and connectivity "
            "to collect real-time information. Smart homes industrial "
            "monitoring and connected healthcare use IoT technology."
        ),
        (
            "Database Management Systems",
            "Database systems organize store and retrieve structured "
            "information. SQL indexing transactions normalization "
            "and query optimization support reliable applications."
        ),
        (
            "Software Development Practices",
            "Software engineering uses version control testing "
            "code reviews continuous integration and deployment. "
            "Agile development helps teams deliver reliable software."
        ),
        (
            "Application Programming Interfaces",
            "APIs allow software applications to communicate. "
            "REST endpoints authentication data formats and "
            "rate limits are important parts of API development."
        ),
        (
            "Digital Marketing Analytics",
            "Digital marketing analytics measures website traffic "
            "campaign engagement audience behaviour and conversion. "
            "Data visualization helps teams evaluate campaigns."
        ),
        (
            "Customer Experience Research",
            "Customer experience research studies satisfaction "
            "feedback product usage and user journeys. Text analysis "
            "can reveal recurring customer concerns."
        ),
        (
            "Supply Chain Technology",
            "Supply chain technology uses forecasting inventory "
            "tracking logistics data and optimization systems "
            "to improve operational visibility."
        ),
        (
            "Predictive Maintenance Systems",
            "Predictive maintenance uses equipment sensor data "
            "and machine learning to anticipate failures. "
            "Condition monitoring can reduce downtime."
        )
    ]

    return pd.DataFrame(
        documents,
        columns=["title", "text"]
    ).assign(
        source="Built-in demo"
    )


# ==========================================================
# TEXT UTILITIES
# ==========================================================

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_snippet(text, length=260):
    text = clean_text(text)

    if len(text) <= length:
        return text

    return text[:length].rsplit(" ", 1)[0] + "..."


# ==========================================================
# TOP HEADER
# ==========================================================

st.markdown("""
<div class="topline">
    <div class="brand">INDEX / KNOWLEDGE SYSTEMS</div>
    <div class="top-meta">DAY 05 OF 60 &nbsp; / &nbsp; NLP RESEARCH DESK</div>
</div>

<div class="hero-label">DOCUMENT INTELLIGENCE / DISCOVERY ENGINE</div>

<div class="hero-title">
    Find the signal<br>
    hidden in your <span>documents.</span>
</div>

<div class="hero-description">
    Search across document collections, uncover related content
    and discover recurring themes using natural language processing
    and machine learning.
</div>
""", unsafe_allow_html=True)

st.write("")


# ==========================================================
# SIDEBAR — INGESTION
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">INDEX / CONTROL</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Document collection")

    source_mode = st.radio(
        "Choose a source",
        ["Demo collection", "Upload documents"],
        label_visibility="collapsed"
    )

    uploaded_files = []

    if source_mode == "Upload documents":

        uploaded_files = st.file_uploader(
            "Upload TXT or CSV files",
            type=["txt", "csv"],
            accept_multiple_files=True
        )

        st.caption(
            "TXT files are treated as individual documents. "
            "For CSV, choose a text column below."
        )

    st.divider()

    st.markdown("### Discovery settings")

    topic_count = st.slider(
        "Number of topics",
        min_value=2,
        max_value=8,
        value=4
    )

    cluster_count_setting = st.slider(
        "Document groups",
        min_value=2,
        max_value=8,
        value=4
    )

    st.divider()

    st.caption("PROCESSING ENGINE")
    st.caption("TF-IDF / COSINE SIMILARITY / NMF / K-MEANS")


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

if source_mode == "Demo collection":

    docs = create_demo_documents().copy()

else:

    if not uploaded_files:
        st.info(
            "Your document desk is ready. Upload TXT or CSV files "
            "from the left panel to begin."
        )
        st.stop()

    loaded = []

    for file in uploaded_files:

        try:

            if file.name.lower().endswith(".txt"):

                content = file.getvalue().decode(
                    "utf-8",
                    errors="ignore"
                )

                loaded.append({
                    "title": file.name,
                    "text": content,
                    "source": file.name
                })

            elif file.name.lower().endswith(".csv"):

                csv_df = pd.read_csv(file)

                if csv_df.empty:
                    continue

                text_candidates = [
                    col for col in csv_df.columns
                    if csv_df[col].dtype == "object"
                ]

                if not text_candidates:
                    st.warning(
                        f"No text column found in {file.name}."
                    )
                    continue

                text_col = st.selectbox(
                    f"Text column for {file.name}",
                    text_candidates,
                    key=f"text_{file.name}"
                )

                title_candidates = [
                    col for col in csv_df.columns
                    if col != text_col
                ]

                title_col = st.selectbox(
                    f"Title column for {file.name}",
                    ["Use row number"] + title_candidates,
                    key=f"title_{file.name}"
                )

                for index, row in csv_df.iterrows():

                    title = (
                        f"Document {index + 1}"
                        if title_col == "Use row number"
                        else str(row[title_col])
                    )

                    loaded.append({
                        "title": title,
                        "text": str(row[text_col]),
                        "source": file.name
                    })

        except Exception as error:

            st.error(
                f"Could not process {file.name}: {error}"
            )

    docs = pd.DataFrame(loaded)

    if docs.empty:
        st.error("No readable documents were found.")
        st.stop()


# ==========================================================
# VALIDATE DOCUMENTS
# ==========================================================

docs = docs.copy()

docs["title"] = docs["title"].fillna("Untitled document").astype(str)
docs["text"] = docs["text"].fillna("").apply(clean_text)
docs["source"] = docs["source"].fillna("Uploaded file")

docs = docs[docs["text"].str.len() > 0].reset_index(drop=True)

if len(docs) < 2:
    st.error("At least two non-empty documents are required.")
    st.stop()


# ==========================================================
# TF-IDF ENGINE
# ==========================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2),
    min_df=1
)

try:

    tfidf_matrix = vectorizer.fit_transform(
        docs["text"]
    )

except ValueError:

    st.error(
        "The documents do not contain enough usable text. "
        "Upload longer documents or use the demo collection."
    )
    st.stop()


if tfidf_matrix.shape[1] < 2:
    st.error(
        "Not enough unique terms were found for document analysis."
    )
    st.stop()


# ==========================================================
# SIMILARITY + TOPIC MODELLING
# ==========================================================

similarity_matrix = cosine_similarity(tfidf_matrix)

actual_topics = min(
    topic_count,
    len(docs) - 1,
    tfidf_matrix.shape[1] - 1
)

actual_topics = max(2, actual_topics)

nmf_model = NMF(
    n_components=actual_topics,
    random_state=42,
    init="nndsvda",
    max_iter=400
)

topic_matrix = nmf_model.fit_transform(tfidf_matrix)

topic_terms = vectorizer.get_feature_names_out()

topic_rows = []

for topic_index, weights in enumerate(nmf_model.components_):

    top_indices = weights.argsort()[::-1][:8]

    terms = [
        topic_terms[index]
        for index in top_indices
    ]

    topic_rows.append({
        "Topic": f"Topic {topic_index + 1}",
        "Keywords": ", ".join(terms)
    })


topic_table = pd.DataFrame(topic_rows)

dominant_topic = topic_matrix.argmax(axis=1)

docs["Topic"] = [
    f"Topic {x + 1}"
    for x in dominant_topic
]

docs["Topic confidence"] = (
    topic_matrix.max(axis=1)
    / np.maximum(topic_matrix.sum(axis=1), 1e-10)
).round(3)


# ==========================================================
# DOCUMENT CLUSTERING
# ==========================================================

actual_clusters = min(
    cluster_count_setting,
    len(docs)
)

cluster_model = KMeans(
    n_clusters=actual_clusters,
    random_state=42,
    n_init=10
)

docs["Cluster"] = cluster_model.fit_predict(
    tfidf_matrix
)


# ==========================================================
# NAVIGATION
# ==========================================================

page = st.radio(
    "Workspace",
    [
        "01  Research Desk",
        "02  Smart Search",
        "03  Topic Discovery",
        "04  Document Library"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# ==========================================================
# PAGE 1 — RESEARCH DESK
# ==========================================================

if page == "01  Research Desk":

    st.markdown("""
    <div class="section-label">01 / COLLECTION OVERVIEW</div>
    <div class="section-title">The research desk.</div>
    <div class="section-description">
        An overview of your document collection and its content structure.
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)

    a.metric("DOCUMENTS", f"{len(docs):,}")
    b.metric("UNIQUE TERMS", f"{tfidf_matrix.shape[1]:,}")
    c.metric("TOPICS FOUND", actual_topics)
    d.metric("DOCUMENT GROUPS", actual_clusters)

    st.write("")

    left, right = st.columns([1.25, 1])

    with left:

        st.markdown("### Recently indexed")

        for _, row in docs.head(5).iterrows():

            st.markdown(
                f"""
                <div class="document-card">
                    <small>{row['source']}</small>
                    <strong>{row['title']}</strong>
                    <p>{make_snippet(row['text'], 170)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with right:

        st.markdown("### Collection composition")

        topic_distribution = (
            docs["Topic"]
            .value_counts()
            .rename_axis("Topic")
            .reset_index(name="Documents")
        )

        fig = px.pie(
            topic_distribution,
            names="Topic",
            values="Documents",
            hole=0.62,
            template="plotly_white",
            color_discrete_sequence=[
                "#315bea",
                "#7894ed",
                "#a9b8ee",
                "#d2d9f4",
                "#5671b9",
                "#9ba8c8"
            ]
        )

        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#fffefa",
            showlegend=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.markdown("### Most frequent terms")

    term_weights = np.asarray(
        tfidf_matrix.sum(axis=0)
    ).ravel()

    feature_names = vectorizer.get_feature_names_out()

    top_indices = term_weights.argsort()[::-1][:15]

    terms_df = pd.DataFrame({
        "Term": feature_names[top_indices],
        "TF-IDF weight": term_weights[top_indices].round(3)
    })

    fig = px.bar(
        terms_df,
        x="TF-IDF weight",
        y="Term",
        orientation="h",
        template="plotly_white",
        color="TF-IDF weight",
        color_continuous_scale=["#a9b8ee", "#315bea"]
    )

    fig.update_layout(
        height=450,
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="#fffefa",
        plot_bgcolor="#fffefa",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# PAGE 2 — SMART SEARCH
# ==========================================================

elif page == "02  Smart Search":

    st.markdown("""
    <div class="section-label">02 / INFORMATION RETRIEVAL</div>
    <div class="section-title">Ask your documents.</div>
    <div class="section-description">
        Search by meaning-related vocabulary and retrieve the
        most similar documents using TF-IDF and cosine similarity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="search-panel">',
        unsafe_allow_html=True
    )

    query = st.text_input(
        "Search your collection",
        placeholder="e.g. machine learning applications in healthcare..."
    )

    result_limit = st.slider(
       "Maximum results",
        min_value=1,
        max_value=min(15, len(docs)),
        value=min(5, len(docs))
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if query.strip():

        query_vector = vectorizer.transform([query])

        if query_vector.nnz == 0:

            st.warning(
                "No searchable terms were found in your query. "
                "Try using different words."
            )

        else:

            scores = cosine_similarity(
                query_vector,
                tfidf_matrix
            ).ravel()

            order = scores.argsort()[::-1][:result_limit]

            search_results = []

            for index in order:

                search_results.append({
                    "Title": docs.iloc[index]["title"],
                    "Source": docs.iloc[index]["source"],
                    "Similarity": round(float(scores[index]), 4),
                    "Topic": docs.iloc[index]["Topic"],
                    "Preview": make_snippet(
                        docs.iloc[index]["text"],
                        220
                    )
                })

            search_df = pd.DataFrame(search_results)

            st.markdown("### Search results")

            st.caption(
                f"Retrieved {len(search_df)} documents "
                f"from a collection of {len(docs)}."
            )

            for result in search_results:

                st.markdown(
                    f"""
                    <div class="document-card">
                        <small>
                            {result['Source']} &nbsp; / &nbsp;
                            {result['Topic']} &nbsp; / &nbsp;
                            SIMILARITY {result['Similarity']:.3f}
                        </small>
                        <strong>{result['Title']}</strong>
                        <p>{result['Preview']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.download_button(
                "Download search results",
                search_df.to_csv(index=False).encode("utf-8"),
                file_name="document_search_results.csv",
                mime="text/csv"
            )

    else:

        st.markdown("### Try a search")

        st.write(
            "Machine learning, cybersecurity, renewable energy, "
            "cloud computing, document search, robotics..."
        )


# ==========================================================
# PAGE 3 — TOPIC DISCOVERY
# ==========================================================

elif page == "03  Topic Discovery":

    st.markdown("""
    <div class="section-label">03 / LATENT THEMES</div>
    <div class="section-title">What are these documents about?</div>
    <div class="section-description">
        Discover recurring themes using Non-negative Matrix Factorization.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Discovered topics")

    for _, row in topic_table.iterrows():

        st.markdown(
            f"""
            <div class="topic-panel">
                <div class="section-label">{row['Topic']}</div>
                <strong style="font-size:17px;color:#17243b;">
                    {row['Keywords']}
                </strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### Topic distribution")

    topic_counts = (
        docs["Topic"]
        .value_counts()
        .rename_axis("Topic")
        .reset_index(name="Documents")
    )

    fig = px.bar(
        topic_counts,
        x="Topic",
        y="Documents",
        text="Documents",
        template="plotly_white",
        color="Documents",
        color_continuous_scale=["#a9b8ee", "#315bea"]
    )

    fig.update_layout(
        height=370,
        paper_bgcolor="#fffefa",
        plot_bgcolor="#fffefa",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.markdown("### Documents assigned to each topic")

    selected_topic = st.selectbox(
        "Choose a topic",
        sorted(docs["Topic"].unique())
    )

    topic_docs = docs[
        docs["Topic"] == selected_topic
    ][["title", "source", "Topic confidence", "text"]].copy()

    topic_docs["text"] = topic_docs["text"].apply(
        lambda x: make_snippet(x, 220)
    )

    st.dataframe(
        topic_docs,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PAGE 4 — DOCUMENT LIBRARY
# ==========================================================

elif page == "04  Document Library":

    st.markdown("""
    <div class="section-label">04 / COLLECTION ARCHIVE</div>
    <div class="section-title">Browse the library.</div>
    <div class="section-description">
        Inspect individual documents, their topics and their
        machine-learning cluster assignments.
    </div>
    """, unsafe_allow_html=True)

    filter_topic = st.selectbox(
        "Filter by topic",
        ["All topics"] + sorted(docs["Topic"].unique())
    )

    filtered = docs.copy()

    if filter_topic != "All topics":

        filtered = filtered[
            filtered["Topic"] == filter_topic
        ]

    st.caption(
        f"Showing {len(filtered)} of {len(docs)} documents."
    )

    st.dataframe(
        filtered[
            [
                "title",
                "source",
                "Topic",
                "Topic confidence",
                "Cluster"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Document inspection")

    selected_title = st.selectbox(
        "Select a document",
        filtered["title"].tolist()
    )

    selected_doc = filtered[
        filtered["title"] == selected_title
    ].iloc[0]

    st.markdown(
        f"""
        <div class="document-card">
            <small>{selected_doc['source']}</small>
            <strong>{selected_doc['title']}</strong>
            <p>
                TOPIC: {selected_doc['Topic']}
                &nbsp; / &nbsp;
                CLUSTER: {selected_doc['Cluster']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.text_area(
        "Document content",
        value=selected_doc["text"],
        height=230,
        disabled=True
    )

    # Find most similar documents
    selected_index = selected_doc.name

    similarity_scores = similarity_matrix[selected_index].copy()
    similarity_scores[selected_index] = -1

    similar_indices = similarity_scores.argsort()[::-1][:5]

    st.markdown("### Related documents")

    related = []

    for index in similar_indices:

        related.append({
            "Document": docs.iloc[index]["title"],
            "Similarity": round(
                float(similarity_scores[index]),
                4
            )
        })

    st.dataframe(
        pd.DataFrame(related),
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# GLOBAL FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
    INDEX / DOCUMENT DISCOVERY ENGINE
    <br><br>
    NATURAL LANGUAGE PROCESSING &nbsp; • &nbsp;
    INFORMATION RETRIEVAL &nbsp; • &nbsp;
    MACHINE LEARNING
    <br><br>
    PROJECT MARATHON / 05 OF 60
</div>
""", unsafe_allow_html=True)
