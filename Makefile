streamlit:
	streamlit run streamlit_app.py --server.enableCORS=false

api:
	uvicorn api:app --reload
