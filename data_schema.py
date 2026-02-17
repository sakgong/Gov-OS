"""
Gov-OS Data Schema and Constants
데이터 표준화 스펙 및 상수 정의

이 파일은 문서에서 지적된 "스케일/단위 불일치" 문제를 해결합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class PolicyStatus(Enum):
    """정책 상태"""
    SUBMITTED = "제출됨"
    REVIEWING = "검토중"
    PASSED_GATE = "게이트통과"
    REJECTED_HARD = "하드제약위반"
    REJECTED_ALIGNMENT = "정렬실패"
    ACCEPTED = "채택됨"
    ARCHIVED = "보관됨"


class RiskLevel(Enum):
    """위험도 수준"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class PolicyInput:
    """
    정책 입력 데이터 표준 스키마
    
    모든 값은 0~1 스케일로 정규화됨 (문서의 스케일 불일치 문제 해결)
    """
    policy_id: str
    title: str
    description: str
    submitter_id: str
    
    # 핵심 지표 (0~1 스케일, 필수)
    R: float  # 난이도 (Resource/Difficulty)
    V: float  # 실측 성과량 (Value)
    ASS: float  # 실행 가능성 (Achievability/Stability/Safety)
    EDI: float  # 형평성/다양성/포용성
    
    # 선택 지표 (0~1 스케일, 결측치 허용)
    innovation_score: Optional[float] = None  # 혁신성
    carbon_impact: Optional[float] = None  # 탄소 영향 (0=나쁨, 1=좋음)
    regional_balance: Optional[float] = None  # 지역 균형

    # 슬라이드 정합성 지표 (0~1 스케일, 선택)
    conflict_index: Optional[float] = None  # 사회적 갈등/충돌지수 (0=낮음, 1=높음)
    fiscal_bust_probability: Optional[float] = None  # 재정 파산/지속불가능 확률 (0~1)
    
    # 예산 및 자원 (실제 값, 양수)
    budget_required: float = 0.0  # 필요 예산 (단위: 백만원)
    budget_available: float = 0.0  # 가용 예산 (단위: 백만원)
    human_resources: int = 0  # 필요 인력 (명)
    
    # 위험 관리
    risk_level: RiskLevel = RiskLevel.MEDIUM
    has_safety_plan: bool = False
    has_legal_review: bool = False
    
    # 메타데이터
    department: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None


@dataclass
class ProfileWeights:
    """
    이해관계자 가중치 프로파일
    
    문서에서 강조: "이건 철학이자 공격면" - 투명하게 공개되어야 함
    """
    version: str = "1.0.0"
    
    # 이해관계자별 가중치 (합=1.0)
    citizen_weight: float = 0.4  # 시민
    expert_weight: float = 0.3  # 전문가
    government_weight: float = 0.3  # 정부
    
    # 가치 차원별 가중치
    efficiency_weight: float = 0.3  # 효율성
    equity_weight: float = 0.3  # 형평성
    innovation_weight: float = 0.2  # 혁신성
    sustainability_weight: float = 0.2  # 지속가능성
    
    def validate(self) -> bool:
        """가중치 합이 1.0인지 검증"""
        stakeholder_sum = (self.citizen_weight + 
                          self.expert_weight + 
                          self.government_weight)
        value_sum = (self.efficiency_weight + 
                    self.equity_weight + 
                    self.innovation_weight + 
                    self.sustainability_weight)
        
        return (abs(stakeholder_sum - 1.0) < 0.01 and 
                abs(value_sum - 1.0) < 0.01)


# 시스템 상수 (문서 기반)
class Constants:
    """Gov-OS 시스템 상수"""
    
    # Layer 1: Hard Constraints
    MIN_BUDGET_COVERAGE = 0.8  # 예산의 최소 80% 확보 필요
    MAX_BUDGET_RATIO = 1.2  # 예산 초과 허용치 20%
    
    # Layer 2: Alignment Gates (Sigmoid 임계값)
    GATE_THRESHOLD = 0.5  # 게이트 통과 기준
    GATE_SIGMOID_STEEPNESS = 10.0  # Sigmoid 가파름 정도
    
    # Layer 3: Core Scoring
    R_AVG = 0.5  # 평균 난이도 (정규화 기준)
    BOOST_CAP = 0.15  # 혁신 보너스 상한 (문서: 독주 방지)
    LEARNING_COEFFICIENT = 0.1  # 학습 가치 계수
    
    # 정규화 안정성
    NORMALIZATION_METHOD = "robust"  # robust/minmax/zscore
    OUTLIER_QUANTILE = 0.95  # 이상치 제거 임계값
    
    # 스팸/중복 감지
    SIMILARITY_THRESHOLD = 0.85  # 중복 판정 유사도
    MAX_SUBMISSIONS_PER_USER_PER_DAY = 10
    
    # 클램핑 범위 (문서: 정의역 보호)
    MIN_SCORE_VALUE = 0.0
    MAX_SCORE_VALUE = 1.0
    EPSILON = 1e-8  # 로그 함수 보호용


# 에러 메시지
class ErrorMessages:
    """사용자 친화적 에러 메시지"""
    
    BUDGET_EXCEEDED = "예산 한도를 초과했습니다. 가용 예산을 확인해주세요."
    MISSING_DATA = "필수 입력 항목이 누락되었습니다: {field}"
    INVALID_RANGE = "{field} 값은 {min}~{max} 범위여야 합니다. 현재값: {value}"
    DATA_CORRUPTION = "데이터 무결성 검증 실패. 관리자에게 문의하세요."
    LEGAL_REVIEW_REQUIRED = "법적 검토가 필요한 정책입니다."
    SAFETY_PLAN_REQUIRED = "안전 계획 수립이 필요합니다."
    DUPLICATE_DETECTED = "유사한 정책이 이미 제출되었습니다."
    SPAM_DETECTED = "스팸으로 의심되는 제출입니다."
