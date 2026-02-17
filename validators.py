"""
Gov-OS Input Validators
입력 검증 및 경계 케이스 처리

문서에서 지적된 문제들 해결:
1. 정의역/클램프 누락 → 모든 입력 클램핑
2. 스케일 불일치 → 강제 0~1 정규화
3. 결측치/이상치 → 명확한 처리 규칙
"""

import math
import hashlib
from typing import Tuple, Optional, List
from data_schema import (
    PolicyInput, Constants, ErrorMessages, 
    RiskLevel, ProfileWeights
)


class ValidationError(Exception):
    """검증 실패 예외"""
    pass


class InputValidator:
    """입력 데이터 검증 및 정제"""
    
    def __init__(self):
        self.submission_history: List[str] = []  # 스팸 감지용
    
    def validate_and_sanitize(self, policy: PolicyInput) -> Tuple[PolicyInput, List[str]]:
        """
        정책 입력 검증 및 정제
        
        Returns:
            (정제된 정책, 경고 메시지 리스트)
        
        Raises:
            ValidationError: 치명적 검증 실패 시
        """
        warnings = []
        
        # 1. 필수 필드 검증
        self._validate_required_fields(policy)
        
        # 2. 범위 검증 및 클램핑 (문서: 정의역 보호)
        policy, range_warnings = self._clamp_values(policy)
        warnings.extend(range_warnings)
        
        # 3. 예산 검증 (Hard Constraint)
        self._validate_budget(policy)
        
        # 4. 논리적 일관성 검증
        self._validate_logical_consistency(policy)
        
        # 5. 스팸/중복 감지
        spam_warning = self._detect_spam(policy)
        if spam_warning:
            warnings.append(spam_warning)
        
        return policy, warnings
    
    def _validate_required_fields(self, policy: PolicyInput):
        """필수 필드 존재 확인"""
        required = ['policy_id', 'title', 'R', 'V', 'ASS', 'EDI']
        
        for field in required:
            value = getattr(policy, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValidationError(
                    ErrorMessages.MISSING_DATA.format(field=field)
                )
    
    def _clamp_values(self, policy: PolicyInput) -> Tuple[PolicyInput, List[str]]:
        """
        모든 수치를 안전한 범위로 클램핑
        
        문서 문제 해결: "ln(1+ASS_eff) 같은 로그 항은 ASS_eff ≥ 0 보장 필요"
        """
        warnings = []
        
        # 0~1 범위 필드들
        fields_0_1 = ['R', 'V', 'ASS', 'EDI', 
                      'innovation_score', 'carbon_impact', 'regional_balance',
                      'conflict_index', 'fiscal_bust_probability']
        
        for field in fields_0_1:
            value = getattr(policy, field, None)
            if value is not None:
                original = value
                clamped = self._safe_clamp(value, 0.0, 1.0)
                
                if abs(original - clamped) > 0.001:
                    warnings.append(
                        f"{field} 값이 범위를 벗어나 조정됨: {original:.3f} → {clamped:.3f}"
                    )
                
                setattr(policy, field, clamped)
        
        # 양수 필드들 (예산, 인력)
        if policy.budget_required < 0:
            warnings.append("필요 예산이 음수여서 0으로 조정됨")
            policy.budget_required = 0.0
        
        if policy.budget_available < 0:
            warnings.append("가용 예산이 음수여서 0으로 조정됨")
            policy.budget_available = 0.0
        
        if policy.human_resources < 0:
            warnings.append("필요 인력이 음수여서 0으로 조정됨")
            policy.human_resources = 0
        
        return policy, warnings
    
    @staticmethod
    def _safe_clamp(value: float, min_val: float, max_val: float) -> float:
        """
        안전한 클램핑 (NaN, Inf 처리 포함)
        
        문서: "정의역/클램프 누락 시 즉시 폭발"
        """
        if math.isnan(value) or math.isinf(value):
            return min_val  # 기본값으로 복구
        
        return max(min_val, min(max_val, value))
    
    def _validate_budget(self, policy: PolicyInput):
        """
        예산 Hard Constraint 검증
        
        문서: Layer 1 - "예산 초과 시 즉시 탈락"
        """
        if policy.budget_required <= 0:
            return  # 예산 불필요한 정책
        
        if policy.budget_available <= 0:
            raise ValidationError(ErrorMessages.BUDGET_EXCEEDED)
        
        coverage = policy.budget_available / policy.budget_required
        
        if coverage < Constants.MIN_BUDGET_COVERAGE:
            raise ValidationError(
                f"예산 부족: 필요 예산의 {coverage*100:.1f}%만 확보됨 "
                f"(최소 {Constants.MIN_BUDGET_COVERAGE*100:.0f}% 필요)"
            )
        
        if coverage > Constants.MAX_BUDGET_RATIO:
            raise ValidationError(
                f"예산 초과: 필요 예산 대비 {coverage*100:.1f}% 확보됨 "
                f"(최대 {Constants.MAX_BUDGET_RATIO*100:.0f}% 허용)"
            )
    
    def _validate_logical_consistency(self, policy: PolicyInput):
        """논리적 일관성 검증"""
        
        # 고위험 정책인데 안전 계획 없으면 경고
        if policy.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            if not policy.has_safety_plan:
                raise ValidationError(ErrorMessages.SAFETY_PLAN_REQUIRED)
        
        # 예산 규모가 큰데 법적 검토 없으면 경고
        if policy.budget_required > 1000:  # 10억 이상
            if not policy.has_legal_review:
                raise ValidationError(ErrorMessages.LEGAL_REVIEW_REQUIRED)
    
    def _detect_spam(self, policy: PolicyInput) -> Optional[str]:
        """
        스팸 및 중복 제출 감지
        
        문서: "스팸/중복 폭주가 실전 가장 흔한 장애 포인트"
        """
        # 정책 해시 생성
        content_hash = self._generate_content_hash(policy)
        
        # 동일 해시 중복 확인
        if content_hash in self.submission_history:
            return ErrorMessages.DUPLICATE_DETECTED
        
        # 히스토리 저장 (최근 1000건만)
        self.submission_history.append(content_hash)
        if len(self.submission_history) > 1000:
            self.submission_history = self.submission_history[-1000:]
        
        return None
    
    @staticmethod
    def _generate_content_hash(policy: PolicyInput) -> str:
        """정책 내용 해시 생성"""
        content = f"{policy.title}|{policy.description}|{policy.submitter_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ProfileValidator:
    """프로파일 가중치 검증"""
    
    @staticmethod
    def validate_profile(profile: ProfileWeights) -> List[str]:
        """
        프로파일 가중치 검증
        
        문서: "가중치는 철학이자 공격면 → 변경 로그 필수"
        """
        warnings = []
        
        if not profile.validate():
            raise ValidationError(
                "프로파일 가중치 합이 1.0이 아닙니다. "
                "이해관계자 또는 가치 차원 가중치를 확인하세요."
            )
        
        # 극단적 가중치 경고
        all_weights = [
            profile.citizen_weight, profile.expert_weight, 
            profile.government_weight,
            profile.efficiency_weight, profile.equity_weight,
            profile.innovation_weight, profile.sustainability_weight
        ]
        
        for weight in all_weights:
            if weight < 0.05:
                warnings.append(f"가중치가 매우 낮습니다 ({weight:.2f}). 의도한 설정인지 확인하세요.")
            if weight > 0.7:
                warnings.append(f"가중치가 매우 높습니다 ({weight:.2f}). 균형을 고려하세요.")
        
        return warnings


# 편의 함수
def safe_log(x: float, base: float = math.e) -> float:
    """
    안전한 로그 함수
    
    문서: "ln(1+ASS_eff) → ASS_eff < 0이면 NaN"
    """
    # epsilon으로 보호
    safe_x = max(Constants.EPSILON, x)
    return math.log(safe_x, base)


def safe_exp(x: float, cap: float = 10.0) -> float:
    """
    안전한 지수 함수 (overflow 방지)
    
    문서: "exp(R/Ravg)는 폭주 가능성"
    """
    # 너무 큰 값 제한
    safe_x = min(x, cap)
    return math.exp(safe_x)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    if abs(denominator) < Constants.EPSILON:
        return default
    return numerator / denominator
