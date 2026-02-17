"""
Gov-OS Core Orchestrator
전체 시스템 통합 및 워크플로우 관리

문서: "Core가 호출/순서/상태/감사로그 책임, 개별 엔진은 계산만"
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

from data_schema import PolicyInput, ProfileWeights, PolicyStatus
from validators import InputValidator, ProfileValidator
from normalizers import NormalizationEngine
from scoring_engine import ScoringEngine, ScoringResult


class GovOSCore:
    """
    Gov-OS 중앙 Orchestrator
    
    책임:
    1. 워크플로우 순서 제어
    2. 상태 관리
    3. 감사 로그
    4. 엔진 간 조율
    """
    
    def __init__(self, profile: ProfileWeights = None, scoring_mode: str = "engine"):
        """
        시스템 초기화
        
        Args:
            profile: 이해관계자 가중치 프로파일 (기본값 사용 가능)
        """
        # 프로파일 검증
        self.profile = profile or ProfileWeights()
        profile_warnings = ProfileValidator.validate_profile(self.profile)
        if profile_warnings:
            print(f"⚠️  프로파일 경고: {profile_warnings}")
        
        # 엔진 초기화
        self.validator = InputValidator()
        self.normalizer = NormalizationEngine()
        self.scorer = ScoringEngine(self.profile, scoring_mode=scoring_mode)
        
        # 감사 로그
        self.audit_log: List[Dict] = []
        
        # 정책 저장소 (실전에서는 DB)
        self.policy_db: Dict[str, Dict] = {}
    
    def process_policy(self, policy: PolicyInput) -> Tuple[ScoringResult, Dict]:
        """
        정책 처리 메인 워크플로우
        
        문서에서 고정된 실행 순서:
        sanitize → validate → score → log
        
        Returns:
            (ScoringResult, audit_record)
        """
        start_time = datetime.now()
        
        # 감사 기록 초기화
        audit_record = {
            'policy_id': policy.policy_id,
            'timestamp': start_time.isoformat(),
            'workflow_steps': [],
            'profile_version': self.profile.version
        }
        
        try:
            # ===== Step 1: Sanitize & Validate =====
            policy, warnings = self.validator.validate_and_sanitize(policy)
            audit_record['workflow_steps'].append({
                'step': 'validate',
                'status': 'success',
                'warnings': warnings
            })
            
            # ===== Step 2: Score =====
            result = self.scorer.score_policy(policy)
            audit_record['workflow_steps'].append({
                'step': 'score',
                'status': 'success',
                'final_score': result.final_score,
                'decision': 'accepted' if result.is_accepted else 'rejected'
            })
            
            # ===== Step 3: Update Status =====
            status = self._determine_status(result)
            audit_record['final_status'] = status.value
            
            # ===== Step 4: Store & Log =====
            self._store_policy(policy, result, status)
            self._log_audit(audit_record)
            
            # 처리 시간 기록
            end_time = datetime.now()
            audit_record['processing_time_ms'] = (
                (end_time - start_time).total_seconds() * 1000
            )
            
            return result, audit_record
        
        except Exception as e:
            # 오류 로깅
            audit_record['workflow_steps'].append({
                'step': 'error',
                'status': 'failed',
                'error': str(e)
            })
            audit_record['final_status'] = PolicyStatus.REJECTED_HARD.value
            self._log_audit(audit_record)
            
            raise
    
    def batch_process(self, policies: List[PolicyInput]) -> List[Tuple[ScoringResult, Dict]]:
        """
        배치 처리 (여러 정책 동시 처리)
        
        장점: 정규화 통계 한 번에 계산 → 안정성 향상
        """
        results = []
        
        # 배치 정규화를 위한 데이터 수집
        all_R = [p.R for p in policies]
        all_V = [p.V for p in policies]
        all_ASS = [p.ASS for p in policies]
        
        # 정규화 통계 계산 (캐싱)
        self.normalizer.normalize_batch(all_R, 'R')
        self.normalizer.normalize_batch(all_V, 'V')
        self.normalizer.normalize_batch(all_ASS, 'ASS')
        
        # 개별 처리
        for policy in policies:
            try:
                result, audit = self.process_policy(policy)
                results.append((result, audit))
            except Exception as e:
                print(f"❌ 정책 {policy.policy_id} 처리 실패: {e}")
                continue
        
        return results
    
    def _determine_status(self, result: ScoringResult) -> PolicyStatus:
        """점수 결과로부터 정책 상태 결정"""
        if not result.passed_hard_constraints:
            return PolicyStatus.REJECTED_HARD
        elif not result.passed_alignment:
            return PolicyStatus.REJECTED_ALIGNMENT
        elif result.is_accepted:
            return PolicyStatus.ACCEPTED
        else:
            return PolicyStatus.REVIEWING
    
    def _store_policy(self, policy: PolicyInput, 
                     result: ScoringResult, status: PolicyStatus):
        """정책 저장 (실전에서는 DB 저장)"""
        self.policy_db[policy.policy_id] = {
            'policy': asdict(policy),
            'result': asdict(result),
            'status': status.value,
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_audit(self, audit_record: Dict):
        """
        감사 로그 기록
        
        문서: "감사 가능성을 시스템 레벨에서 강제"
        """
        # 로그 해시 생성 (변조 방지)
        log_str = json.dumps(audit_record, sort_keys=True)
        audit_record['hash'] = hashlib.sha256(log_str.encode()).hexdigest()[:16]
        
        self.audit_log.append(audit_record)
    
    # ========== 조회 및 분석 ==========
    
    def get_policy_status(self, policy_id: str) -> Optional[Dict]:
        """정책 상태 조회"""
        return self.policy_db.get(policy_id)
    
    def get_audit_trail(self, policy_id: str) -> List[Dict]:
        """특정 정책의 감사 추적"""
        return [
            log for log in self.audit_log 
            if log['policy_id'] == policy_id
        ]
    
    def generate_statistics(self) -> Dict:
        """
        전체 통계 생성
        
        Returns:
            - 총 정책 수
            - 상태별 분포
            - 평균 점수
            - 처리 시간 통계
        """
        if not self.policy_db:
            return {'total': 0}
        
        statuses = [p['status'] for p in self.policy_db.values()]
        scores = [p['result']['final_score'] for p in self.policy_db.values()]
        times = [log.get('processing_time_ms', 0) for log in self.audit_log]
        
        return {
            'total_policies': len(self.policy_db),
            'status_distribution': {
                status: statuses.count(status) 
                for status in set(statuses)
            },
            'score_statistics': {
                'mean': sum(scores) / len(scores) if scores else 0,
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0
            },
            'performance': {
                'avg_processing_time_ms': sum(times) / len(times) if times else 0,
                'total_audits': len(self.audit_log)
            }
        }
    
    def export_audit_log(self, filepath: str = 'audit_log.json'):
        """감사 로그 내보내기"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.audit_log, f, ensure_ascii=False, indent=2)
        
        return f"감사 로그 {len(self.audit_log)}건을 {filepath}에 저장했습니다."
    
    def get_transparency_report(self, policy_id: str) -> Dict:
        """
        투명성 보고서 생성
        
        문서: "투명성 강화 - 알고리즘 공개 + 외부 검증"
        """
        policy_data = self.policy_db.get(policy_id)
        if not policy_data:
            return {'error': '정책을 찾을 수 없습니다'}
        
        result = ScoringResult(**policy_data['result'])
        
        # 민감도 분석
        policy_obj = PolicyInput(**policy_data['policy'])
        sensitivity = self.scorer.sensitivity_analysis(policy_obj)
        
        return {
            'policy_id': policy_id,
            'final_score': result.final_score,
            'decision': 'accepted' if result.is_accepted else 'rejected',
            
            # 점수 분해 (투명성)
            'score_breakdown': {
                'core_score': result.core_score,
                'boost_score': result.boost_score,
                'learning_value': result.learning_value
            },
            
            # 게이트 상세
            'gate_details': result.gate_scores,
            
            # 가중치 (공개)
            'profile_weights': {
                'citizen': self.profile.citizen_weight,
                'expert': self.profile.expert_weight,
                'government': self.profile.government_weight
            },
            
            # 민감도 분석
            'sensitivity_analysis': sensitivity,
            
            # 설명
            'decision_trace': result.decision_trace
        }


# ========== 편의 함수 ==========

def create_default_system() -> GovOSCore:
    """기본 설정으로 시스템 생성"""
    return GovOSCore()


def quick_score(policy_dict: Dict) -> float:
    """빠른 점수 계산 (딕셔너리 입력)"""
    policy = PolicyInput(**policy_dict)
    system = create_default_system()
    result, _ = system.process_policy(policy)
    return result.final_score
