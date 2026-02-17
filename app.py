import sys
from pathlib import Path
import uuid
from datetime import datetime
from io import BytesIO

import streamlit as st

# --- 엔진 모듈 경로 추가 ---
ENGINE_DIR = Path(__file__).parent / "engine"
sys.path.insert(0, str(ENGINE_DIR))

from data_schema import PolicyInput, ProfileWeights, RiskLevel  # type: ignore
from gov_os_core import GovOSCore  # type: ignore


# ------------------------------
# 페이지 설정
# ------------------------------
st.set_page_config(
    page_title="Gov-OS 정책평가시스템(데모)",
    page_icon="🏛️",
    layout="wide",
)


# ------------------------------
# 정부 스타일 CSS
# ------------------------------
NAVY = "#0F172A"
BLUE = "#1E40AF"      # 정책정합도
GREEN = "#16A34A"     # 안정성
YELLOW = "#FACC15"    # 사회갈등
RED = "#DC2626"       # 평가중단
TEXT = "#FFFFFF"
MUTED = "#94A3B8"

st.markdown(
    f"""
<style>
/* Base */
.block-container {{ padding-top: 1.2rem; padding-bottom: 2.0rem; }}

.gov-header {{
  background: linear-gradient(135deg, {NAVY} 0%, #111C3A 60%, #0B1224 100%);
  border-radius: 14px;
  padding: 18px 20px;
  color: {TEXT};
  border: 1px solid rgba(148,163,184,0.18);
}}
.gov-title {{ font-size: 26px; font-weight: 800; margin: 0; line-height: 1.2; }}
.gov-subtitle {{ margin: 6px 0 0 0; color: {MUTED}; font-size: 14px; }}

/* Cards */
.kpi-wrap {{
  border-radius: 16px;
  padding: 18px 18px;
  background: #F8FAFC;
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
}}
.kpi-label {{ font-size: 13px; color: #334155; margin: 0 0 8px 0; font-weight: 700; }}
.kpi-num {{ font-size: 56px; font-weight: 900; margin: 0; line-height: 1.0; }}
.kpi-unit {{ font-size: 40px; font-weight: 800; opacity: 0.95; }}
.kpi-desc {{ font-size: 12px; color: #64748B; margin: 10px 0 0 0; }}

.kpi-fit .kpi-num {{ color: {BLUE}; }}
.kpi-safety .kpi-num {{ color: {GREEN}; }}
.kpi-conflict .kpi-num {{ color: #B45309; }}

/* Status */
.status-box {{
  border-radius: 14px;
  padding: 14px 16px;
  border: 1px solid rgba(15,23,42,0.08);
  background: #FFFFFF;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
}}
.status-title {{ font-size: 14px; margin: 0; color: #334155; font-weight: 800; }}
.status-value {{ font-size: 22px; margin: 4px 0 0 0; font-weight: 900; }}
.status-ok {{ color: {GREEN}; }}
.status-review {{ color: {RED}; }}
.status-stop {{ color: {RED}; }}

/* Small helpers */
.small-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; color: #475569; }}

</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------
# 표준 행정용어 매핑
# ------------------------------
MODE_LABEL = {
    "hybrid": "혼합(보수적)",
    "demo": "표준식(3지표)",
    "engine": "엔진식(고급)",
}

RISK_LABEL = {
    "LOW": "낮음(1)",
    "MEDIUM": "보통(2)",
    "HIGH": "높음(3)",
    "VERY_HIGH": "매우 높음(4)",
    "CRITICAL": "심각(5)",
}


def _pct(x: float | None) -> float:
    if x is None:
        return 0.0
    return max(0.0, min(100.0, float(x) * 100.0))


def _fmt_pct(num: float, digits: int = 1) -> str:
    return f"{num:.{digits}f}%"


def _normalize_triplet(a: float, b: float, c: float):
    s = a + b + c
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)


def _normalize_quad(a: float, b: float, c: float, d: float):
    s = a + b + c + d
    if s <= 0:
        return (0.25, 0.25, 0.25, 0.25)
    return (a / s, b / s, c / s, d / s)


# ------------------------------
# 헤더
# ------------------------------
st.markdown(
    f"""
<div class="gov-header">
  <div class="gov-title">Gov-OS 정책평가시스템 (데모)</div>
  <div class="gov-subtitle">정책 결정의 재현성과 투명성을 위한 정량평가 · 자동기록 · 보고서 생성</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# ------------------------------
# 사이드바: 설정/프로파일
# ------------------------------
with st.sidebar:
    st.header("설정")
    scoring_mode = st.selectbox(
        "평가모드",
        options=["hybrid", "demo", "engine"],
        index=0,
        format_func=lambda m: MODE_LABEL.get(m, m),
        help="혼합(보수적): 엔진/표준식 중 낮은 값 선택 · 표준식: 3지표 수식 · 엔진식: 고도화 공식",
    )

    st.subheader("평가 프로파일(가중치)")
    st.caption("합이 1이 되도록 자동 정규화합니다.")

    st.markdown("**이해관계자 가중치**")
    c_w = st.slider("국민 관점", 0.0, 1.0, 0.4, 0.01)
    e_w = st.slider("전문가 관점", 0.0, 1.0, 0.3, 0.01)
    g_w = st.slider("정부(집행) 관점", 0.0, 1.0, 0.3, 0.01)

    st.markdown("**가치 기준 가중치**")
    eff_w = st.slider("효율성", 0.0, 1.0, 0.30, 0.01)
    eq_w = st.slider("형평성", 0.0, 1.0, 0.30, 0.01)
    inn_w = st.slider("혁신성", 0.0, 1.0, 0.20, 0.01)
    sus_w = st.slider("지속가능성", 0.0, 1.0, 0.20, 0.01)

    c_w, e_w, g_w = _normalize_triplet(c_w, e_w, g_w)
    eff_w, eq_w, inn_w, sus_w = _normalize_quad(eff_w, eq_w, inn_w, sus_w)

    profile = ProfileWeights(
        version="2.0.3-demo",
        citizen_weight=float(c_w),
        expert_weight=float(e_w),
        government_weight=float(g_w),
        efficiency_weight=float(eff_w),
        equity_weight=float(eq_w),
        innovation_weight=float(inn_w),
        sustainability_weight=float(sus_w),
    )

    st.divider()
    st.subheader("데모 시나리오(원클릭)")
    st.caption("영상 촬영/설명용 입력값 세트")


# ------------------------------
# Core 초기화
# ------------------------------
if "core" not in st.session_state or st.session_state.get("_mode") != scoring_mode or st.session_state.get("_profile") != profile:
    st.session_state["core"] = GovOSCore(profile=profile, scoring_mode=scoring_mode)
    st.session_state["_mode"] = scoring_mode
    st.session_state["_profile"] = profile

core: GovOSCore = st.session_state["core"]


# ------------------------------
# PDF 보고서 생성
# ------------------------------
def generate_pdf_report(policy: PolicyInput, result, audit: dict) -> bytes:
    # reportlab로 간단 브리핑 PDF 생성
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Gov-OS 정책평가 브리핑 보고서")
    y -= 24

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"생성시각: {datetime.now().isoformat(timespec='seconds')}")
    y -= 18
    c.drawString(40, y, f"시스템 버전: Gov-OS v2.0.3 | 평가모드: {MODE_LABEL.get(scoring_mode, scoring_mode)}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "1) 정책안 개요")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"정책안 식별번호: {policy.policy_id}")
    y -= 14
    c.drawString(50, y, f"정책안 명칭: {policy.title}")
    y -= 14
    if getattr(policy, "department", ""):
        c.drawString(50, y, f"소관부서: {policy.department}")
        y -= 14
    c.drawString(50, y, f"예산(요구/가용): {policy.budget_required:.1f} / {policy.budget_available:.1f} (백만원)")
    y -= 14
    c.drawString(50, y, f"위험등급: {RISK_LABEL.get(policy.risk_level.name, policy.risk_level.name)}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "2) 핵심 지표")
    y -= 16
    c.setFont("Helvetica", 10)

    fit = _pct(getattr(result, "fit_score", 0.0))
    safety = _pct(getattr(result, "safety_score", 0.0))
    conflict = _pct(getattr(result, "conflict_score", 0.0))
    final = _pct(getattr(result, "final_score", 0.0))

    c.drawString(50, y, f"정책정합도(Fit): {fit:.1f}%")
    y -= 14
    c.drawString(50, y, f"안정성지수(Safety): {safety:.1f}%")
    y -= 14
    c.drawString(50, y, f"사회갈등지수(Conflict): {conflict:.1f}%")
    y -= 14
    c.drawString(50, y, f"최종 평가점수: {final:.1f}%")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "3) 판정")
    y -= 16
    c.setFont("Helvetica", 10)

    if not getattr(result, "passed_hard_constraints", True):
        decision_text = "평가중단(하드 제약 위반)"
    else:
        decision_text = "적정" if getattr(result, "is_accepted", False) else "재검토"
    c.drawString(50, y, f"최종 판정: {decision_text}")
    y -= 14

    # HardFail 근거/경고
    warnings = getattr(result, "warnings", []) or []
    if warnings:
        c.drawString(50, y, "유의사항/경고:")
        y -= 14
        for w in warnings[:6]:
            c.drawString(60, y, f"- {str(w)}")
            y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "4) 감사 기록 요약")
    y -= 16
    c.setFont("Helvetica", 10)

    c.drawString(50, y, f"감사기록 해시: {audit.get('hash', '')}")
    y -= 14
    c.drawString(50, y, f"처리시간(ms): {audit.get('processing_time_ms', '')}")
    y -= 14
    c.drawString(50, y, f"최종 상태: {audit.get('final_status', '')}")

    c.showPage()
    c.save()

    return buf.getvalue()


# ------------------------------
# 입력 폼
# ------------------------------
def build_policy_form(seed: str = "") -> PolicyInput:
    col1, col2, col3 = st.columns([1.25, 1.0, 1.0])

    with col1:
        policy_id = st.text_input("정책안 식별번호", value=seed or f"P-{uuid.uuid4().hex[:8]}")
        title = st.text_input("정책안 명칭", value="정책안 A")
        description = st.text_area("정책안 개요", value="정책안 개요를 입력하십시오.", height=120)
        submitter_id = st.text_input("작성자 식별자", value="u-001")
        department = st.text_input("소관부서", value="")
        tags_str = st.text_input("키워드(쉼표 구분)", value="")

    with col2:
        st.markdown("#### 평가 입력(0~1)")
        R = st.slider("난이도", 0.0, 1.0, 0.50, 0.01)
        V = st.slider("기대성과", 0.0, 1.0, 0.60, 0.01)
        ASS = st.slider("실행가능성", 0.0, 1.0, 0.70, 0.01)
        EDI = st.slider("형평·포용", 0.0, 1.0, 0.50, 0.01)

        st.markdown("#### 선택 입력(0~1)")
        innovation_score = st.slider("혁신성", 0.0, 1.0, 0.50, 0.01)
        carbon_impact = st.slider("탄소영향(0 나쁨 ~ 1 좋음)", 0.0, 1.0, 0.50, 0.01)
        regional_balance = st.slider("지역균형", 0.0, 1.0, 0.50, 0.01)

    with col3:
        st.markdown("#### 준수/위험")
        risk_name = st.selectbox(
            "위험등급",
            options=[rl.name for rl in RiskLevel],
            index=[rl.name for rl in RiskLevel].index("MEDIUM"),
            format_func=lambda x: RISK_LABEL.get(x, x),
        )
        has_safety_plan = st.checkbox("안전대책 수립", value=True)
        has_legal_review = st.checkbox("법령 검토 완료", value=True)

        st.markdown("#### 슬라이드 정합성(선택)")
        conflict_index = st.slider("사회갈등지수(0 낮음 ~ 1 높음)", 0.0, 1.0, 0.32, 0.01)
        fiscal_bust_probability = st.slider("재정지속위험확률(0~1)", 0.0, 1.0, 0.10, 0.01)

        st.markdown("#### 예산/자원")
        budget_required = st.number_input("요구예산(백만원)", min_value=0.0, value=1000.0, step=100.0)
        budget_available = st.number_input("가용예산(백만원)", min_value=0.0, value=1200.0, step=100.0)
        human_resources = st.number_input("필요인력(명)", min_value=0, value=5, step=1)

    tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    return PolicyInput(
        policy_id=policy_id,
        title=title,
        description=description,
        submitter_id=submitter_id,
        R=float(R),
        V=float(V),
        ASS=float(ASS),
        EDI=float(EDI),
        innovation_score=float(innovation_score),
        carbon_impact=float(carbon_impact),
        regional_balance=float(regional_balance),
        budget_required=float(budget_required),
        budget_available=float(budget_available),
        human_resources=int(human_resources),
        risk_level=RiskLevel[risk_name],
        has_safety_plan=bool(has_safety_plan),
        has_legal_review=bool(has_legal_review),
        department=department,
        tags=tags,
        created_at=datetime.now().isoformat(timespec="seconds"),
        conflict_index=float(conflict_index),
        fiscal_bust_probability=float(fiscal_bust_probability),
    )


# ------------------------------
# 3지표 카드 렌더
# ------------------------------
def render_three_kpis(result):
    fit = _pct(getattr(result, "fit_score", 0.0))
    safety = _pct(getattr(result, "safety_score", 0.0))
    conflict = _pct(getattr(result, "conflict_score", 0.0))

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
<div class="kpi-wrap kpi-fit">
  <div class="kpi-label">정책정합도</div>
  <div class="kpi-num">{fit:.0f}<span class="kpi-unit">%</span></div>
  <div class="kpi-desc">정책 목표와의 정합성</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
<div class="kpi-wrap kpi-safety">
  <div class="kpi-label">안정성지수</div>
  <div class="kpi-num">{safety:.0f}<span class="kpi-unit">%</span></div>
  <div class="kpi-desc">법적·재정 리스크 수준</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
<div class="kpi-wrap kpi-conflict">
  <div class="kpi-label">사회갈등지수</div>
  <div class="kpi-num">{conflict:.0f}<span class="kpi-unit">%</span></div>
  <div class="kpi-desc">사회적 갈등 가능성</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_decision_box(result, audit: dict):
    # 판정 로직: 하드 제약 불통과면 평가중단, 그 외 accept면 적정, 아니면 재검토
    if not getattr(result, "passed_hard_constraints", True):
        decision = "평가중단"
        cls = "status-stop"
        subtitle = "하드 제약(중단 조건) 위반"
    else:
        if getattr(result, "is_accepted", False):
            decision = "적정"
            cls = "status-ok"
            subtitle = "정책 추진 적정"
        else:
            decision = "재검토"
            cls = "status-review"
            subtitle = "보완 후 재평가 권고"

    final = _pct(getattr(result, "final_score", 0.0))

    st.markdown(
        f"""
<div class="status-box">
  <div class="status-title">최종 판정</div>
  <div class="status-value {cls}">{decision}</div>
  <div class="kpi-desc">{subtitle}</div>
  <div style="height:10px;"></div>
  <div class="status-title">최종 평가점수</div>
  <div class="status-value">{final:.1f}%</div>
  <div class="kpi-desc">동일 입력이면 동일 결과 · Gov-OS v2.0.3</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(f"감사기록 해시: {audit.get('hash','')} | 최종상태: {audit.get('final_status','')} | 처리시간(ms): {audit.get('processing_time_ms','')}")


# ------------------------------
# 탭
# ------------------------------
tab_eval, tab_batch, tab_audit, tab_help = st.tabs([
    "정책안 평가",
    "다건 비교",
    "감사기록",
    "도움말",
])


# ------------------------------
# 탭: 정책안 평가
# ------------------------------
with tab_eval:
    st.subheader("정책안 평가")
    st.caption("표준행정용어 적용 · 정부 스타일 UI · 3지표(정합/안정/갈등) · 평가중단(하드 제약) · 보고서 생성")

    # 데모 시나리오 버튼
    s1, s2, s3 = st.columns(3)
    if s1.button("✅ 시나리오 A: 적정(안정)"):
        st.session_state["seed"] = "SCN-A"
        st.session_state["preset"] = "A"
    if s2.button("⛔ 시나리오 B: 평가중단(재정)"):
        st.session_state["seed"] = "SCN-B"
        st.session_state["preset"] = "B"
    if s3.button("⚠️ 시나리오 C: 평가중단(갈등)"):
        st.session_state["seed"] = "SCN-C"
        st.session_state["preset"] = "C"

    preset = st.session_state.get("preset")
    seed = st.session_state.get("seed", "")

    policy = build_policy_form(seed=seed)

    # 프리셋 적용(사용자가 버튼 클릭 시)
    if preset == "A":
        policy.V = 0.80
        policy.R = 0.55
        policy.ASS = 0.85
        policy.EDI = 0.70
        policy.conflict_index = 0.20
        policy.fiscal_bust_probability = 0.10
        policy.risk_level = RiskLevel.MEDIUM
        policy.has_safety_plan = True
        policy.has_legal_review = True
    elif preset == "B":
        policy.V = 0.65
        policy.R = 0.60
        policy.ASS = 0.70
        policy.EDI = 0.55
        policy.conflict_index = 0.30
        policy.fiscal_bust_probability = 0.45
        policy.risk_level = RiskLevel.HIGH
        policy.has_safety_plan = True
        policy.has_legal_review = True
    elif preset == "C":
        policy.V = 0.70
        policy.R = 0.55
        policy.ASS = 0.75
        policy.EDI = 0.55
        policy.conflict_index = 0.85
        policy.fiscal_bust_probability = 0.10
        policy.risk_level = RiskLevel.MEDIUM
        policy.has_safety_plan = True
        policy.has_legal_review = True

    run = st.button("정책평가 실행", type="primary")

    if run:
        try:
            result, audit = core.process_policy(policy)

            st.success("평가가 완료되었습니다.")

            # 핵심 화면: 3지표 카드 + 수식
            render_three_kpis(result)
            st.write("")
            st.markdown("**평가 수식(표준식):**  Final = Fit × Safety × (1 - Conflict)")

            # 판정/점수 박스
            st.write("")
            render_decision_box(result, audit)

            st.divider()
            cL, cR = st.columns([1, 1])

            with cL:
                st.subheader("자동 브리핑 보고서")
                pdf_bytes = generate_pdf_report(policy, result, audit)
                st.download_button(
                    "보고서(PDF) 다운로드",
                    data=pdf_bytes,
                    file_name=f"GovOS_정책평가_보고서_{policy.policy_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.caption("국회·감사 대응용 요약(데모) — 실전 PoC에서는 서식/결재라인을 추가합니다.")

            with cR:
                st.subheader("감사기록(상세)")
                st.json(audit)

            with st.expander("세부 산출(엔진 내부)"):
                st.json({
                    "final_score": result.final_score,
                    "engine_score": getattr(result, "engine_score", None),
                    "demo_score": getattr(result, "demo_score", None),
                    "formula_used": getattr(result, "formula_used", None),
                    "core_score": result.core_score,
                    "boost_score": result.boost_score,
                    "learning_value": result.learning_value,
                    "gate_scores": result.gate_scores,
                    "warnings": result.warnings,
                    "decision_trace": result.decision_trace,
                })

        except Exception as e:
            st.error(f"평가 처리 중 오류: {e}")


# ------------------------------
# 탭: 다건 비교
# ------------------------------
with tab_batch:
    st.subheader("다건 비교(배치 평가)")
    st.caption("여러 정책안을 동시에 비교하여 상대평가/우선순위를 검토합니다.")

    n = st.slider("정책안 개수", 2, 30, 7)
    seed_conflict = st.checkbox("사회갈등지수 분산", value=True)

    if st.button("배치 평가 실행"):
        policies: list[PolicyInput] = []
        for i in range(n):
            p = build_policy_form(seed=f"B-{i+1:03d}")
            # 작은 변동(데모용)
            p.V = max(0.0, min(1.0, p.V + (i - n / 2) * 0.02))
            p.R = max(0.0, min(1.0, p.R + (n / 2 - i) * 0.015))
            if seed_conflict:
                p.conflict_index = max(0.0, min(1.0, (i / max(1, n - 1))))
            policies.append(p)

        try:
            results = core.batch_process(policies)
            rows = []
            for r, a in results:
                rows.append({
                    "정책안 식별번호": a["policy_id"],
                    "최종 평가점수": round(_pct(r.final_score), 1),
                    "정책정합도": round(_pct(getattr(r, "fit_score", 0.0)), 1),
                    "안정성지수": round(_pct(getattr(r, "safety_score", 0.0)), 1),
                    "사회갈등지수": round(_pct(getattr(r, "conflict_score", 0.0)), 1),
                    "판정": "적정" if r.is_accepted else "재검토" if r.passed_hard_constraints else "평가중단",
                    "최종상태": a.get("final_status"),
                })

            st.dataframe(rows, use_container_width=True)

        except Exception as e:
            st.error(f"배치 평가 오류: {e}")


# ------------------------------
# 탭: 감사기록
# ------------------------------
with tab_audit:
    st.subheader("감사기록")
    st.caption("평가 이력(데모)은 메모리 저장소에 저장됩니다. 실전 PoC에서는 DB/로그 스토리지를 연동합니다.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 정책안 저장 현황")
        st.write(f"총 {len(core.policy_db)}건")
        if core.policy_db:
            st.dataframe(
                [
                    {
                        "정책안 식별번호": k,
                        "상태": v["status"],
                        "최종점수": round(_pct(v["result"]["final_score"]), 1),
                        "시각": v["timestamp"],
                    }
                    for k, v in core.policy_db.items()
                ],
                use_container_width=True,
            )

    with c2:
        st.markdown("#### 감사로그(최근 50건)")
        st.write(f"총 {len(core.audit_log)}건")
        if core.audit_log:
            st.dataframe(
                [
                    {
                        "정책안 식별번호": l["policy_id"],
                        "시각": l["timestamp"],
                        "해시": l.get("hash"),
                        "최종상태": l.get("final_status"),
                    }
                    for l in core.audit_log[-50:]
                ],
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### 정책안별 감사 추적")
    pid = st.text_input("정책안 식별번호로 조회")
    if st.button("조회") and pid:
        trail = core.get_audit_trail(pid)
        if not trail:
            st.warning("해당 정책안의 감사기록이 없습니다.")
        else:
            st.json(trail)


# ------------------------------
# 탭: 도움말
# ------------------------------
with tab_help:
    st.subheader("실행 안내")
    st.markdown("**설치**")
    st.code("py -m pip install -r requirements.txt", language="bash")
    st.markdown("**실행**")
    st.code("py -m streamlit run app.py", language="bash")

    st.divider()
    st.subheader("표준행정용어 적용 범위")
    st.write(
        "- 정책(policy) → 정책안\n"
        "- 평가점수(score) → 평가점수\n"
        "- HardFail → 평가중단\n"
        "- Fit/Safety/Conflict → 정책정합도/안정성지수/사회갈등지수\n"
        "- PASS/FAIL/ACCEPT/REJECT → 적정/재검토/평가중단(표시 방식)"
    )

    st.divider()
    st.subheader("운영 전환 시 체크리스트")
    st.write(
        "- 사용자 인증/권한(작성자·검토자·관리자)\n"
        "- 결재/검토 단계(제출→검토→확정) 상태전이\n"
        "- DB/로그 스토리지 연동(감사 대응)\n"
        "- 보고서 서식(기관 로고/문서번호/결재라인) 적용\n"
        "- 기준값(평가중단 임계치) 정책별/기관별 프로파일로 분리"
    )

