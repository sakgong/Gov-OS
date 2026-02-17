"""
Gov-OS Scoring Engine
핵심 점수 계산 엔진 (3-Layer 구조)

문서 구조 구현:
Layer 1: Hard Constraint (즉시 탈락)
Layer 2: Alignment Pass (게이트 검증)
Layer 3: Core Score + Controlled Boost

추가(정합성 패치):
- 3지표(Fit/Safety/Conflict) 산출
- 데모 수식: Final_demo = Fit × Safety × (1 - Conflict)
- HardFail: 재정 파산 확률/충돌지수 임계치
- scoring_mode: engine | demo | hybrid
"""

import math
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from data_schema import PolicyInput, ProfileWeights, Constants
from validators import safe_exp


@dataclass
class ScoringResult:
    """점수 계산 결과"""

    # 최종 점수(현재 선택된 모드 결과)
    final_score: float  # 0~1

    # Layer 결과
    passed_hard_constraints: bool
    passed_alignment: bool
    alignment_score: float

    # 엔진(기존) 구성요소
    core_score: float
    boost_score: float
    learning_value: float

    # 세부 게이트 점수
    gate_scores: Dict[str, float]

    # 3지표(슬라이드 정합성)
    fit_score: float
    safety_score: float
    conflict_score: float

    # 모드별 점수(투명성)
    engine_score: float
    demo_score: float
    formula_used: str  # engine|demo|hybrid

    # 설명
    decision_trace: str
    warnings: list

    @property
    def is_accepted(self) -> bool:
        return (
            self.passed_hard_constraints
            and self.passed_alignment
            and self.final_score > 0
        )


class ScoringEngine:
    """
    Gov-OS 점수 계산 엔진

    - engine 모드: 문서 엔진 수식 유지
    - demo 모드: 영상/피치용 3지표 곱셈 수식
    - hybrid 모드: 과대평가 방지(보수적) → min(engine, demo)
    """

    def __init__(self, profile: ProfileWeights = None, scoring_mode: str = "engine"):
        self.profile = profile or ProfileWeights()
        if not self.profile.validate():
            raise ValueError("Invalid profile weights")

        scoring_mode = (scoring_mode or "engine").lower().strip()
        if scoring_mode not in {"engine", "demo", "hybrid"}:
            raise ValueError("scoring_mode must be one of: engine, demo, hybrid")
        self.scoring_mode = scoring_mode

    def score_policy(self, policy: PolicyInput) -> ScoringResult:
        warnings = []

        # ===== Layer 1: Hard Constraints =====
        hard_check, hard_msg = self._check_hard_constraints(policy)
        if not hard_check:
            return ScoringResult(
                final_score=0.0,
                passed_hard_constraints=False,
                passed_alignment=False,
                alignment_score=0.0,
                core_score=0.0,
                boost_score=0.0,
                learning_value=0.0,
                gate_scores={},
                fit_score=0.0,
                safety_score=0.0,
                conflict_score=0.0,
                engine_score=0.0,
                demo_score=0.0,
                formula_used=self.scoring_mode,
                decision_trace=f"하드 제약 위반: {hard_msg}",
                warnings=warnings,
            )

        # ===== Layer 2: Alignment Pass =====
        gate_scores = self._calculate_gate_scores(policy)
        alignment_score = self._aggregate_gates(gate_scores)
        passed_alignment = alignment_score >= Constants.GATE_THRESHOLD

        # 슬라이드용 3지표(게이트/입력 기반)
        fit, safety, conflict = self._calculate_three_metrics(policy, gate_scores)
        demo_score = float(np.clip(fit * safety * (1.0 - conflict), 0.0, 1.0))

        if not passed_alignment:
            return ScoringResult(
                final_score=0.0,
                passed_hard_constraints=True,
                passed_alignment=False,
                alignment_score=float(alignment_score),
                core_score=0.0,
                boost_score=0.0,
                learning_value=0.0,
                gate_scores=gate_scores,
                fit_score=float(fit),
                safety_score=float(safety),
                conflict_score=float(conflict),
                engine_score=0.0,
                demo_score=demo_score,
                formula_used=self.scoring_mode,
                decision_trace=self._explain_alignment_failure(gate_scores),
                warnings=warnings,
            )

        # ===== Layer 3: Core Score + Boost =====
        core = self._calculate_core_score(policy)
        boost = self._calculate_boost_score(policy)
        learning = self._calculate_learning_value(policy)

        engine_score = core * (1.0 + boost) + Constants.LEARNING_COEFFICIENT * learning
        engine_score = float(np.clip(engine_score, 0.0, 1.0))

        # 모드에 따라 최종 점수 결정
        if self.scoring_mode == "engine":
            final_score = engine_score
            formula_used = "engine"
        elif self.scoring_mode == "demo":
            final_score = demo_score
            formula_used = "demo"
        else:
            final_score = min(engine_score, demo_score)
            formula_used = "hybrid"

        return ScoringResult(
            final_score=float(final_score),
            passed_hard_constraints=True,
            passed_alignment=True,
            alignment_score=float(alignment_score),
            core_score=float(core),
            boost_score=float(boost),
            learning_value=float(learning),
            gate_scores=gate_scores,
            fit_score=float(fit),
            safety_score=float(safety),
            conflict_score=float(conflict),
            engine_score=engine_score,
            demo_score=demo_score,
            formula_used=formula_used,
            decision_trace=self._explain_success(core, boost, learning, engine_score, demo_score, final_score, formula_used, fit, safety, conflict),
            warnings=warnings,
        )

    # ========== Layer 1: Hard Constraints ==========

    def _check_hard_constraints(self, policy: PolicyInput) -> Tuple[bool, str]:
        # (A) HardFail: 재정 파산 확률
        if policy.fiscal_bust_probability is not None:
            if policy.fiscal_bust_probability >= 0.45:
                return False, f"HardFail: 재정 파산 확률 {policy.fiscal_bust_probability:.0%} ≥ 45%"

        # (B) HardFail: 충돌지수(사회 갈등)
        conflict = self._calculate_conflict_index(policy)
        if conflict >= 0.85:
            return False, f"HardFail: 충돌지수 {conflict:.0%} ≥ 85%"

        # 안전성 검증
        if policy.risk_level.value >= 4:  # HIGH, VERY_HIGH
            if not policy.has_safety_plan:
                return False, "고위험 정책에 안전 계획 부재"

        # 법적 검토 필수
        if policy.budget_required > 1000:  # 10억 이상 (단위: 백만원)
            if not policy.has_legal_review:
                return False, "대규모 예산 정책에 법적 검토 부재"

        # 핵심 지표 무결성
        critical_fields = [policy.R, policy.V, policy.ASS, policy.EDI]
        if any(x < 0 or x > 1 or math.isnan(x) for x in critical_fields):
            return False, "핵심 지표 데이터 손상"

        return True, ""

    # ========== Layer 2: Alignment Pass ==========

    def _calculate_gate_scores(self, policy: PolicyInput) -> Dict[str, float]:
        gates: Dict[str, float] = {}

        gates["ASS"] = self._sigmoid_gate(policy.ASS, 0.4)

        integrity_score = 0.5
        if policy.has_legal_review:
            integrity_score += 0.25
        if policy.has_safety_plan:
            integrity_score += 0.25
        gates["Gamma"] = self._sigmoid_gate(integrity_score, 0.5)

        phi_score = (
            policy.EDI * self.profile.equity_weight
            + policy.V * self.profile.efficiency_weight
            + (policy.innovation_score or 0.5) * self.profile.innovation_weight
            + (policy.carbon_impact or 0.5) * self.profile.sustainability_weight
        )
        gates["Phi"] = self._sigmoid_gate(phi_score, 0.4)

        risk_normalized = 1.0 - (policy.risk_level.value / 5.0)
        control_score = risk_normalized
        if policy.has_safety_plan:
            control_score = min(1.0, control_score + 0.2)
        gates["RC"] = self._sigmoid_gate(control_score, 0.3)

        return gates

    @staticmethod
    def _sigmoid_gate(score: float, threshold: float = 0.5) -> float:
        k = Constants.GATE_SIGMOID_STEEPNESS
        x_shifted = score - threshold
        exp_term = safe_exp(-k * x_shifted, cap=20)
        return 1.0 / (1.0 + exp_term)

    def _aggregate_gates(self, gate_scores: Dict[str, float]) -> float:
        if not gate_scores:
            return 0.0
        product = 1.0
        for score in gate_scores.values():
            product *= score
        return product

    def _calculate_three_metrics(self, policy: PolicyInput, gate_scores: Dict[str, float]) -> Tuple[float, float, float]:
        fit = float(gate_scores.get("Phi", 0.0))

        safety_components = [
            float(gate_scores.get("ASS", 0.0)),
            float(gate_scores.get("Gamma", 0.0)),
            float(gate_scores.get("RC", 0.0)),
        ]
        safety = float(np.clip(sum(safety_components) / max(1, len(safety_components)), 0.0, 1.0))

        conflict = float(self._calculate_conflict_index(policy))
        return fit, safety, conflict

    # ========== Conflict Engine (고도화 1차) ==========

    def _calculate_conflict_index(self, policy: PolicyInput) -> float:
        """사회적 갈등/충돌지수 계산(0~1).

        - policy.conflict_index가 있으면 그대로 사용(입력 우선)
        - 없으면, MVP 단계 프록시 모델로 산출
        """
        if policy.conflict_index is not None:
            return float(np.clip(policy.conflict_index, 0.0, 1.0))

        # 프록시 모델: EDI, 위험도, 지역균형, 예산 규모, 태그 신호
        edi_term = (1.0 - float(policy.EDI)) * 0.35

        risk_norm = (float(policy.risk_level.value) - 1.0) / 4.0
        risk_term = risk_norm * 0.25

        if policy.regional_balance is not None:
            regional_term = (1.0 - float(policy.regional_balance)) * 0.20
        else:
            regional_term = 0.10

        budget_term = min(0.20, (float(policy.budget_required) / 50000.0) * 0.20)

        tag_term = 0.0
        if policy.tags:
            lowered = " ".join(policy.tags).lower()
            for kw in ["갈등", "민감", "규제", "세금", "재개발", "반대", "controvers", "tax", "regulat"]:
                if kw in lowered:
                    tag_term = 0.10
                    break

        conflict = 0.10 + edi_term + risk_term + regional_term + budget_term + tag_term
        return float(np.clip(conflict, 0.0, 1.0))

    # ========== Layer 3: Core Score ==========

    def _calculate_core_score(self, policy: PolicyInput) -> float:
        R = policy.R
        V = policy.V
        R_avg = Constants.R_AVG

        difficulty_multiplier = safe_exp(R / (R_avg + Constants.EPSILON), cap=2.0)
        feasibility_factor = policy.ASS
        core = V * difficulty_multiplier * feasibility_factor
        return float(np.clip(core, 0.0, 1.0))

    def _calculate_boost_score(self, policy: PolicyInput) -> float:
        innovation = policy.innovation_score or 0.0
        boost = innovation * Constants.BOOST_CAP
        return float(np.clip(boost, 0.0, Constants.BOOST_CAP))

    def _calculate_learning_value(self, policy: PolicyInput) -> float:
        learning = policy.R * (1.0 - policy.ASS) * 0.5
        return float(np.clip(learning, 0.0, 0.2))

    # ========== 설명 생성 ==========

    def _explain_alignment_failure(self, gate_scores: Dict[str, float]) -> str:
        failed_gates = [name for name, score in gate_scores.items() if score < Constants.GATE_THRESHOLD]
        gate_names = {"ASS": "실행 가능성", "Gamma": "무결성", "Phi": "전략 정합성", "RC": "위험 통제"}
        failed_names = [gate_names.get(g, g) for g in failed_gates]

        return (
            f"정렬 게이트 미통과: {', '.join(failed_names)} 개선 필요. "
            f"전체 정렬 점수: {self._aggregate_gates(gate_scores):.2f} "
            f"(기준: {Constants.GATE_THRESHOLD:.2f})"
        )

    def _explain_success(
        self,
        core: float,
        boost: float,
        learning: float,
        engine_score: float,
        demo_score: float,
        final_score: float,
        formula_used: str,
        fit: float,
        safety: float,
        conflict: float,
    ) -> str:
        return (
            f"채택 권장 (최종: {final_score:.3f}, 모드: {formula_used})\n"
            f"- 엔진 점수(engine): {engine_score:.3f} = {core:.3f} × (1 + {boost:.3f}) + {Constants.LEARNING_COEFFICIENT} × {learning:.3f}\n"
            f"- 데모 점수(demo): {demo_score:.3f} = Fit × Safety × (1 - Conflict)\n"
            f"  · Fit={fit:.3f}, Safety={safety:.3f}, Conflict={conflict:.3f}"
        )

    # ========== 프로파일 민감도 분석 ==========

    def sensitivity_analysis(self, policy: PolicyInput, weight_variation: float = 0.1) -> Dict:
        base_result = self.score_policy(policy)
        variations = {}

        for weight_name in ["citizen_weight", "expert_weight", "government_weight"]:
            original = getattr(self.profile, weight_name)

            setattr(self.profile, weight_name, min(1.0, original * (1.0 + weight_variation)))
            plus_result = self.score_policy(policy)

            setattr(self.profile, weight_name, max(0.0, original * (1.0 - weight_variation)))
            minus_result = self.score_policy(policy)

            setattr(self.profile, weight_name, original)

            variations[weight_name] = {
                "base": base_result.final_score,
                "plus": plus_result.final_score,
                "minus": minus_result.final_score,
                "sensitivity": abs(plus_result.final_score - minus_result.final_score),
            }

        return variations
