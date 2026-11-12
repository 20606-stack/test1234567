import os
import pickle
import numpy as np
import streamlit as st

# Streamlit Cloud(리눅스 서버) 환경에서 Orange3 임포트 시 GUI 충돌 방지
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Orange.data import Domain, Table


# ---------------------------------------------------------
# 1. 페이지 설정 및 커스텀 CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="교통사고 사망자 예측 & 스쿨존 안전 신호등",
    page_icon="🚦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .traffic-box {
        background-color: #1e2229;
        padding: 24px 18px;
        border-radius: 24px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 16px;
    }
    .lamp {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-weight: 800;
        font-size: 18px;
        color: white;
        transition: all 0.25s ease;
    }
    .lamp-off {
        opacity: 0.18;
        filter: grayscale(70%);
        box-shadow: none;
    }
    .lamp-green-on {
        background: radial-gradient(circle at 30% 30%, #4ade80, #16a34a);
        box-shadow: 0 0 30px #22c55e, 0 0 60px rgba(34, 197, 94, 0.5);
        transform: scale(1.06);
    }
    .lamp-yellow-on {
        background: radial-gradient(circle at 30% 30%, #facc15, #ca8a04);
        box-shadow: 0 0 30px #eab308, 0 0 60px rgba(234, 179, 8, 0.5);
        color: #1f2937;
        transform: scale(1.06);
    }
    .lamp-red-on {
        background: radial-gradient(circle at 30% 30%, #f87171, #dc2626);
        box-shadow: 0 0 30px #ef4444, 0 0 60px rgba(239, 68, 68, 0.55);
        transform: scale(1.06);
    }
    .safety-card {
        padding: 20px 22px;
        border-radius: 16px;
        border-left: 8px solid;
        background-color: #f8fafc;
        color: #0f172a;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    }
    .safety-card h4 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 19px;
    }
    .safety-card ul {
        margin-bottom: 0;
        padding-left: 20px;
        line-height: 1.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. 오렌지3 모델 로드 (@st.cache_resource로 1회만 로드)
# ---------------------------------------------------------
MODEL_PATH = "traffic_casualty_model.pkcls"


@st.cache_resource
def load_orange_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    model = load_orange_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"모델 파일(`{MODEL_PATH}`)을 찾을 수 없습니다. GitHub 저장소 루트에 파일을 업로드해주세요.")
    st.stop()
except Exception as e:
    st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()


# ---------------------------------------------------------
# 3. 예측 실행 함수
# ---------------------------------------------------------
def predict_fatalities(
    loaded_model,
    sido: str,
    accidents: float,
    minor_injuries: float,
    severe_injuries: float = np.nan,
    reported_injuries: float = np.nan,
) -> float:
    """
    Orange3 모델의 original_domain에 맞춰 입력 Table을 생성하고 예측값을 반환합니다.
    입력되지 않은 변수(np.nan)는 모델 내부 전처리기(ReplaceUnknowns)가 학습 평균값으로 자동 대체합니다.
    """
    orig_domain = loaded_model.original_domain
    feature_domain = Domain(orig_domain.attributes)

    sido_var = orig_domain["시도"]
    sido_idx = float(sido_var.to_val(sido))

    # 피처 이름 순서에 맞게 입력 배열 매핑
    value_map = {
        "시도": sido_idx,
        "사고건수": float(accidents),
        "중상자수": float(severe_injuries),
        "경상자수": float(minor_injuries),
        "부상신고자수": float(reported_injuries),
    }

    row = [value_map.get(attr.name, np.nan) for attr in orig_domain.attributes]
    X = np.array([row], dtype=float)

    input_table = Table.from_numpy(feature_domain, X)
    prediction = loaded_model(input_table)

    # 회귀 모델 특성상 0 미만 예측값이 나올 경우 0.0으로 보정
    return max(0.0, float(prediction[0]))


# ---------------------------------------------------------
# 4. 사이드바: 위험도 신호등 기준값 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 신호등 판정 기준 설정")
    st.caption("예상 사망자 수에 따른 안전/주의/위험 구간을 조정할 수 있습니다.")
    safe_limit = st.number_input("🟢 안전 상한 (미만)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
    caution_limit = st.number_input("🟡 주의 상한 (미만)", min_value=safe_limit + 0.5, max_value=50.0, value=7.0, step=0.5)


# ---------------------------------------------------------
# 5. 메인 화면 구성: 입력 슬라이더 & 실시간 예측 결과
# ---------------------------------------------------------
st.title("🚦 교통사고 사망자 수 예측 및 어린이 보호구역 안전 대시보드")
st.write("시도와 사고 발생 현황을 조절하면 **Orange3 인공지능 모델**이 예상 사망자 수와 스쿨존 안전 단계를 실시간으로 계산합니다.")

st.divider()

col_input, col_result = st.columns([1.05, 1.15], gap="large")

with col_input:
    st.subheader("1️⃣ 교통사고 조건 입력")

    # 모델에 학습된 시도 목록 자동 추출 (['경기', '부산', '서울'])
    sido_options = list(model.original_domain["시도"].values)
    selected_sido = st.selectbox("📍 시도 선택", options=sido_options, index=0)

    accidents = st.slider(
        "🚗 사고건수 (건)",
        min_value=10,
        max_value=5000,
        value=1200,
        step=10,
    )

    minor_injuries = st.slider(
        "🩹 경상자수 (명)",
        min_value=10,
        max_value=6000,
        value=1300,
        step=10,
    )

    with st.expander("➕ 세부 부상자 수 직접 입력 (선택사항)"):
        use_custom_injuries = st.checkbox("중상자수·부상신고자수 직접 조절하기 (해제 시 학습 평균 자동 적용)", value=False)
        if use_custom_injuries:
            severe_injuries = st.slider("🚑 중상자수 (명)", min_value=0, max_value=2000, value=300, step=5)
            reported_injuries = st.slider("📋 부상신고자수 (명)", min_value=0, max_value=1000, value=150, step=5)
        else:
            severe_injuries = np.nan
            reported_injuries = np.nan

# 슬라이더 조작 시마다 실시간 예측 실행
predicted_deaths = predict_fatalities(
    loaded_model=model,
    sido=selected_sido,
    accidents=accidents,
    minor_injuries=minor_injuries,
    severe_injuries=severe_injuries,
    reported_injuries=reported_injuries,
)

# 단계 판정
if predicted_deaths < safe_limit:
    status = "SAFE"
    status_label = "🟢 안전 단계"
    border_color = "#16a34a"
    card_bg = "#f0fdf4"
    card_title = "🟢 [안전] 어린이 보호구역 기본 안전 수칙"
    safety_rules = [
        "스쿨존 내 제한속도 <b>시속 30km 이하</b> 준수 및 전방 주시 철저",
        "횡단보도 앞 <b>일시정지(보행자 유무와 관계없이)</b> 생활화",
        "어린이는 보행 중 스마트폰 사용을 멈추고 좌우 확인 후 횡단",
        " 불법 주·정차 금지로 어린이 보행자 시야 확보 유지",
    ]
elif predicted_deaths < caution_limit:
    status = "CAUTION"
    status_label = "🟡 주의 단계"
    border_color = "#eab308"
    card_bg = "#fefce8"
    card_title = "🟡 [주의] 어린이 보호구역 강화 안전 수칙"
    safety_rules = [
        "등·하교 시간대 스쿨존 진입 시 <b>시속 20km 이하 서행</b> 및 방어운전",
        "주정차 차량 사이에서 어린이가 갑자기 뛰어드는 돌발 상황 대비",
        "우회전 시 횡단보도 보행자 신호와 사각지대(A필러·차량 하단) 재확인",
        "녹색어머니회·교통안전지도사의 수신호 및 안전 유도 적극 협조",
    ]
else:
    status = "DANGER"
    status_label = "🔴 위험 단계"
    border_color = "#dc2626"
    card_bg = "#fef2f2"
    card_title = "🔴 [위험] 어린이 보호구역 특별 경보 및 필수 수칙"
    safety_rules = [
        "<b>스쿨존 우회 도로 이용 권장</b> 및 불가피한 진입 시 초저속(20km/h 미만) 운행",
        "교차로·횡단보도 진입 전 완전 정지 후 3초 이상 주변 어린이 확인",
        "보호자 동반 등·하교 지도 강화 및 밝은 색 옷·가방 안전 반사경 착용",
        "지자체·경찰 합동 집중 단속 및 스쿨존 불법 주정차 즉시 신고",
    ]

with col_result:
    st.subheader("2️⃣ 실시간 예측 결과 및 신호등")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label=f"{selected_sido} 예상 사망자 수", value=f"{predicted_deaths:.2f} 명")
    with metric_col2:
        st.metric(label="현재 위험도 판정", value=status_label)

    # 대형 3구 신호등 HTML 렌더링
    green_cls = "lamp lamp-green-on" if status == "SAFE" else "lamp lamp-off"
    yellow_cls = "lamp lamp-yellow-on" if status == "CAUTION" else "lamp lamp-off"
    red_cls = "lamp lamp-red-on" if status == "DANGER" else "lamp lamp-off"

    st.markdown(
        f"""
        <div class="traffic-box">
            <div class="{green_cls}" style="background-color:#16a34a;">
                <span>🟢</span>
                <span>안전</span>
            </div>
            <div class="{yellow_cls}" style="background-color:#eab308;">
                <span>🟡</span>
                <span>주의</span>
            </div>
            <div class="{red_cls}" style="background-color:#dc2626;">
                <span>🔴</span>
                <span>위험</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 어린이 보호구역 안전 수칙 카드
    rules_html = "".join([f"<li>{rule}</li>" for rule in safety_rules])
    st.markdown(
        f"""
        <div class="safety-card" style="border-left-color: {border_color}; background-color: {card_bg};">
            <h4>{card_title}</h4>
            <ul>{rules_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )