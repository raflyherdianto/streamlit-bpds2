import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Prediksi Status Mahasiswa JJI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_INPUT_COLUMNS_ORDERED = [
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Scholarship_holder',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_1st_sem_enrolled',
    'Admission_grade',
    'Displaced'
]

PIPELINE_MODEL_PATH = './student_logreg.joblib'

try:
    model_pipeline = joblib.load(PIPELINE_MODEL_PATH)
    print(f"Pipeline model '{PIPELINE_MODEL_PATH}' berhasil dimuat.")
except FileNotFoundError:
    st.error(f"File pipeline model '{PIPELINE_MODEL_PATH}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan app.py dan nama file sudah benar.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat pipeline model: {e}")
    st.stop()

def predict_status_with_pipeline(user_inputs_dict):
    """
    Melakukan prediksi menggunakan pipeline model yang sudah dimuat.
    Input: dictionary dengan kunci adalah nama fitur asli dan value adalah input pengguna.
    Output: array probabilitas prediksi.
    """
    try:
        input_df = pd.DataFrame([user_inputs_dict])
        input_df_ordered = input_df[MODEL_INPUT_COLUMNS_ORDERED]
        prediction_proba = model_pipeline.predict_proba(input_df_ordered)
        return prediction_proba
    except KeyError as ke:
        st.error(f"KeyError saat membuat DataFrame input: {ke}. Pastikan semua fitur di MODEL_INPUT_COLUMNS_ORDERED ada di user_inputs_dict.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi dengan pipeline: {e}")
        import traceback
        st.text(traceback.format_exc()) 
        return None

status_dict = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
status_emoji = {'Dropout': '😞', 'Enrolled': '➡️', 'Graduate': '🎉'}

st.markdown("""
    <style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 20px;
        /* text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3); */
    }
    .sub-header {
        font-size: 20px;
        color: #4682B4;
        text-align: center;
        margin-bottom: 30px;
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out, color 0.2s ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        width: auto;
        min-width: 200px;
        margin: 0 auto;
    }
    div.stButton > button:first-child:hover { 
        background-color: #0056b3;
        color: white;
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    div.stButton > button:first-child:active {
        background-color: #004085; 
        color: white;
        transform: scale(0.98);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .prediction-result {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f8ff;
        margin-top: 20px;
        border: 1px solid #b0c4de;
        text-align: center;
    }
    .prediction-text {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        color: #333333;
        margin-bottom: 1rem;
    }
    .probability-container {
        text-align: center;
        margin-top: 1rem;
    }
    .probability-title {
        font-size: 18px;
        font-weight: bold;
        color: #333333;
        margin-bottom: 0.75rem;
    }
    .probability-list {
        list-style-type: none;
        padding-left: 0;
        margin-top: 0;
        margin-bottom: 0;
        display: inline-block;
        text-align: left;
    }
    .probability-list li {
        font-size: 16px;
        color: #333333;
        margin-bottom: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎓 Prediksi Status Kelulusan Mahasiswa</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Jaya Jaya Institut - Sistem Peringatan Dini</p>', unsafe_allow_html=True)

st.sidebar.header("Masukkan Data Mahasiswa:")

curricular_units_1st_sem_approved_input = st.sidebar.number_input('SKS Lulus Semester 1 (Curricular units 1st sem (approved))', min_value=0, max_value=30, value=5, help="Jumlah SKS yang disetujui di semester 1 (0-30).")
curricular_units_1st_sem_grade_input = st.sidebar.number_input('Rata-rata Nilai Semester 1 (Curricular units 1st sem (grade))', min_value=0.0, max_value=20.0, value=10.0, step=0.1, format="%.1f", help="Nilai rata-rata semester 1 (skala 0-20).")
curricular_units_2nd_sem_approved_input = st.sidebar.number_input('SKS Lulus Semester 2 (Curricular units 2nd sem (approved))', min_value=0, max_value=30, value=5, help="Jumlah SKS yang disetujui di semester 2 (0-30).")
curricular_units_2nd_sem_grade_input = st.sidebar.number_input('Rata-rata Nilai Semester 2 (Curricular units 2nd sem (grade))', min_value=0.0, max_value=20.0, value=10.0, step=0.1, format="%.1f", help="Nilai rata-rata semester 2 (skala 0-20).")
curricular_units_1st_sem_enrolled_input = st.sidebar.number_input('SKS Diambil Semester 1 (Curricular units 1st sem (enrolled))', min_value=0, max_value=30, value=6, help="Jumlah SKS yang diambil di semester 1 (0-30).")
curricular_units_2nd_sem_enrolled_input = st.sidebar.number_input('SKS Diambil Semester 2 (Curricular units 2nd sem (enrolled))', min_value=0, max_value=30, value=6, help="Jumlah SKS yang diambil di semester 2 (0-30).")
admission_grade_input = st.sidebar.slider('Nilai Masuk (Admission grade)', min_value=0.0, max_value=200.0, value=120.0, step=0.1, format="%.1f", help="Nilai saat penerimaan (skala 0-200).")

tuition_options = {1: 'Ya (Lunas)', 0: 'Tidak (Belum Lunas)'}
tuition_input_val = st.sidebar.selectbox('Status Pembayaran UKT (Tuition fees up to date)', options=list(tuition_options.keys()), format_func=lambda x: tuition_options[x])

scholarship_options = {1: 'Ya', 0: 'Tidak'}
scholarship_input_val = st.sidebar.selectbox('Penerima Beasiswa (Scholarship holder)', options=list(scholarship_options.keys()), format_func=lambda x: scholarship_options[x])

displaced_options = {1: 'Ya (Mahasiswa Pindahan)', 0: 'Tidak (Bukan Pindahan)'}
displaced_input_val = st.sidebar.selectbox('Status Pindahan (Displaced)', options=list(displaced_options.keys()), format_func=lambda x: displaced_options[x])

user_data_for_prediction = {
    'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved_input,
    'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade_input,
    'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved_input,
    'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade_input,
    'Tuition_fees_up_to_date': tuition_input_val,
    'Scholarship_holder': scholarship_input_val,
    'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled_input,
    'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled_input,
    'Admission_grade': admission_grade_input,
    'Displaced': displaced_input_val
}

col1, col_button, col3 = st.columns([0.35, 0.3, 0.35])

with col_button:
    predict_button = st.button('Prediksi Status')

if predict_button:
    prediction_proba = predict_status_with_pipeline(user_data_for_prediction)

    if prediction_proba is not None:
        predicted_class_index = np.argmax(prediction_proba[0])
        
        if predicted_class_index not in status_dict:
            st.error(f"Indeks prediksi ({predicted_class_index}) di luar jangkauan status_dict. Harap periksa model dan mapping status.")
        else:
            predicted_status_label = status_dict[predicted_class_index]
            predicted_emoji = status_emoji.get(predicted_status_label, "❓")

            with st.container():
                html_output = f'''
                <div class="prediction-result">
                    <p class="prediction-text">{predicted_emoji} Prediksi Status Mahasiswa: <strong>{predicted_status_label}</strong></p>
                    <hr style="margin-top: 1rem; margin-bottom: 1rem;">
                    <div class="probability-container">
                        <p class="probability-title">Probabilitas:</p>
                        <ul class="probability-list">
                            <li>Dropout: {prediction_proba[0][0]*100:.2f}%</li>
                            <li>Enrolled: {prediction_proba[0][1]*100:.2f}%</li>
                            <li>Graduate: {prediction_proba[0][2]*100:.2f}%</li>
                        </ul>
                    </div>
                </div>
                '''
                st.markdown(html_output, unsafe_allow_html=True)

            st.subheader("Grafik Probabilitas Status")
            status_labels_for_df = [status_dict.get(i, f"Kelas {i}") for i in range(prediction_proba.shape[1])]
            if len(status_labels_for_df) == prediction_proba.shape[1]:
                prob_df = pd.DataFrame({'Status': status_labels_for_df, 'Probabilitas': prediction_proba[0]})
                prob_df = prob_df.set_index('Status')
                st.bar_chart(prob_df)
            else:
                st.warning("Tidak dapat membuat grafik probabilitas karena ketidakcocokan jumlah kelas.")
    else:
        pass

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: #777;">
    Aplikasi Prototipe untuk Prediksi Status Mahasiswa Jaya Jaya Institut.<br>
    Model yang digunakan: Regresi Logistik (dalam Pipeline). Dibuat dengan Streamlit.
</div>
""", unsafe_allow_html=True)