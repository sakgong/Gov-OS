"""
Gov-OS Normalization Engine
안정적인 정규화 엔진

문서 문제 해결: "Min-Max 정규화는 정책 추가/삭제 시 전체 순위 영향"
→ Robust Scaler 사용으로 안정성 확보
"""

import numpy as np
from typing import List, Dict, Tuple
from data_schema import Constants


class NormalizationEngine:
    """
    정규화 엔진
    
    문서 지적: "배치 min-max는 대안 추가/삭제에 민감"
    → Robust Scaler로 이상치 영향 최소화
    """
    
    def __init__(self, method: str = Constants.NORMALIZATION_METHOD):
        """
        Args:
            method: 'robust' (기본, 안정적) / 'minmax' / 'zscore'
        """
        self.method = method
        self.stats_cache: Dict[str, Tuple[float, float]] = {}
    
    def normalize_batch(self, values: List[float], 
                       field_name: str = "unnamed") -> np.ndarray:
        """
        배치 정규화 (여러 정책을 동시에 처리)
        
        Args:
            values: 정규화할 값 리스트
            field_name: 필드명 (통계 캐싱용)
        
        Returns:
            0~1 범위로 정규화된 numpy array
        """
        if not values:
            return np.array([])
        
        arr = np.array(values, dtype=float)
        
        # NaN/Inf 제거
        arr = self._clean_array(arr)
        
        if self.method == "robust":
            return self._robust_normalize(arr, field_name)
        elif self.method == "minmax":
            return self._minmax_normalize(arr, field_name)
        elif self.method == "zscore":
            return self._zscore_normalize(arr, field_name)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def normalize_single(self, value: float, 
                        field_name: str = "unnamed") -> float:
        """
        단일 값 정규화 (캐시된 통계 사용)
        
        실시간 정책 추가 시 사용 - 기존 분포 유지
        """
        if field_name not in self.stats_cache:
            # 통계 없으면 그대로 반환 (클램핑만)
            return np.clip(value, 0.0, 1.0)
        
        arr = np.array([value])
        normalized = self._apply_cached_normalization(arr, field_name)
        return float(normalized[0])
    
    def _robust_normalize(self, arr: np.ndarray, 
                         field_name: str) -> np.ndarray:
        """
        Robust Scaler 정규화
        
        장점: 이상치에 강건, 정책 추가/삭제 시 안정적
        방법: IQR(Interquartile Range) 기반
        """
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        # IQR이 0이면 (모든 값이 같으면) 중간값 사용
        if iqr < Constants.EPSILON:
            median = np.median(arr)
            self.stats_cache[field_name] = (median, 1.0)
            return np.full_like(arr, 0.5)
        
        # (x - Q1) / IQR → [0, 1]로 스케일링
        median = np.median(arr)
        normalized = (arr - median) / (1.5 * iqr)  # 1.5 * IQR로 범위 확장
        
        # 0~1로 클램핑 및 재스케일
        normalized = (normalized + 1) / 2  # [-1, 1] → [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # 통계 캐싱
        self.stats_cache[field_name] = (median, iqr)
        
        return normalized
    
    def _minmax_normalize(self, arr: np.ndarray, 
                         field_name: str) -> np.ndarray:
        """
        Min-Max 정규화 (전통적 방법)
        
        단점: 새로운 정책이 최대/최소값을 갱신하면 전체 순위 변동
        """
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        
        if range_val < Constants.EPSILON:
            self.stats_cache[field_name] = (min_val, 1.0)
            return np.full_like(arr, 0.5)
        
        normalized = (arr - min_val) / range_val
        
        # 통계 캐싱
        self.stats_cache[field_name] = (min_val, max_val)
        
        return normalized
    
    def _zscore_normalize(self, arr: np.ndarray, 
                         field_name: str) -> np.ndarray:
        """
        Z-Score 정규화
        
        장점: 정규분포 가정 시 해석 용이
        단점: 이상치에 민감
        """
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std < Constants.EPSILON:
            self.stats_cache[field_name] = (mean, 1.0)
            return np.full_like(arr, 0.5)
        
        # z = (x - mean) / std
        z_scores = (arr - mean) / std
        
        # [-3, 3] 범위를 [0, 1]로 매핑
        normalized = (z_scores + 3) / 6
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # 통계 캐싱
        self.stats_cache[field_name] = (mean, std)
        
        return normalized
    
    def _apply_cached_normalization(self, arr: np.ndarray, 
                                    field_name: str) -> np.ndarray:
        """캐시된 통계로 정규화 (실시간 추가 정책용)"""
        stat1, stat2 = self.stats_cache[field_name]
        
        if self.method == "robust":
            median, iqr = stat1, stat2
            normalized = (arr - median) / (1.5 * iqr)
            normalized = (normalized + 1) / 2
        elif self.method == "minmax":
            min_val, max_val = stat1, stat2
            normalized = (arr - min_val) / (max_val - min_val)
        elif self.method == "zscore":
            mean, std = stat1, stat2
            z_scores = (arr - mean) / std
            normalized = (z_scores + 3) / 6
        else:
            normalized = arr
        
        return np.clip(normalized, 0.0, 1.0)
    
    @staticmethod
    def _clean_array(arr: np.ndarray) -> np.ndarray:
        """NaN, Inf 제거 및 대체"""
        # NaN을 중앙값으로 대체
        median = np.nanmedian(arr)
        if np.isnan(median):
            median = 0.5
        
        arr = np.where(np.isnan(arr), median, arr)
        
        # Inf를 최대/최소값으로 대체
        valid_values = arr[np.isfinite(arr)]
        if len(valid_values) > 0:
            max_valid = np.max(valid_values)
            min_valid = np.min(valid_values)
            arr = np.where(np.isposinf(arr), max_valid, arr)
            arr = np.where(np.isneginf(arr), min_valid, arr)
        else:
            arr = np.full_like(arr, 0.5)
        
        return arr
    
    def remove_outliers(self, values: List[float], 
                       quantile: float = Constants.OUTLIER_QUANTILE) -> List[float]:
        """
        이상치 제거 (Robust)

        기존 분위수 기반은 소표본에서 경계값이 느슨해질 수 있음.
        → MAD(중앙절대편차) 기반 modified z-score로 강건하게 제거.
        """
        if not values:
            return []

        arr = np.array(values, dtype=float)
        arr = self._clean_array(arr)

        median = np.median(arr)
        abs_dev = np.abs(arr - median)
        mad = np.median(abs_dev)

        # MAD가 0이면(대부분 동일값) 분위수 방식으로 폴백
        if mad < Constants.EPSILON:
            lower = np.percentile(arr, (1 - quantile) * 100 / 2)
            upper = np.percentile(arr, quantile * 100 + (1 - quantile) * 100 / 2)
            filtered = arr[(arr >= lower) & (arr <= upper)]
            return filtered.tolist()

        modified_z = 0.6745 * (arr - median) / mad

        # 임계값은 quantile에 따라 조정(0.95 → 약 3.5)
        # 0.99 → 약 4.5 수준으로 완만히 상승
        threshold = 3.5 + max(0.0, (quantile - 0.95) * 20.0)

        filtered = arr[np.abs(modified_z) <= threshold]
        return filtered.tolist()
    
    def get_stability_report(self) -> Dict[str, Dict]:
        """
        정규화 안정성 보고서
        
        문서 요구사항: "정규화 안정성 테스트 필수"
        """
        report = {}
        
        for field_name, (stat1, stat2) in self.stats_cache.items():
            report[field_name] = {
                "method": self.method,
                "stat1": stat1,
                "stat2": stat2,
                "is_stable": stat2 > Constants.EPSILON
            }
        
        return report
