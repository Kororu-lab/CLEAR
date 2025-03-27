# CLEAR: 클러스터링 및 주가 영향 기반 뉴스 추천 시스템

## 기술 보고서

**날짜:** 2025년 3월 26일  
**버전:** 1.0

## 목차

1. [요약](#요약)
2. [서론](#서론)
3. [시스템 아키텍처](#시스템-아키텍처)
4. [데이터 수집 및 처리](#데이터-수집-및-처리)
   - [뉴스 크롤러](#뉴스-크롤러)
   - [주가 데이터 수집기](#주가-데이터-수집기)
   - [텍스트 전처리기](#텍스트-전처리기)
5. [핵심 알고리즘](#핵심-알고리즘)
   - [뉴스 벡터화](#뉴스-벡터화)
   - [뉴스 클러스터링](#뉴스-클러스터링)
   - [주가 영향 분석](#주가-영향-분석)
   - [뉴스 추천](#뉴스-추천)
6. [파이프라인 통합](#파이프라인-통합)
7. [평가 프레임워크](#평가-프레임워크)
8. [성능 분석](#성능-분석)
9. [배포 및 사용법](#배포-및-사용법)
10. [향후 개선 사항](#향후-개선-사항)
11. [결론](#결론)
12. [참고 문헌](#참고-문헌)

## 요약

CLEAR(Clustering and Stock Impact-based News Recommendation System)는 네이버의 AiRs 아키텍처를 기반으로 한 고급 뉴스 추천 시스템으로, 특별히 금융 뉴스 분석과 주가 영향 평가에 맞게 조정되었습니다. 원래의 AiRs 시스템이 개인화에 중점을 두는 것과 달리, CLEAR는 뉴스가 주가에 미치는 영향을 우선시하여 재무적으로 중요한 뉴스 기사를 강조하는 추천을 제공합니다.

이 시스템은 네이버 AiRs의 모든 핵심 메커니즘(협업 필터링, 콘텐츠 기반 필터링, 품질 평가, 사회적 관심도, 최신 뉴스 우선순위)을 구현하면서 개인화 구성 요소를 주가 영향 분석으로 대체합니다. CLEAR는 한국어 금융 뉴스 기사를 처리하고, 관련 콘텐츠를 클러스터링하며, 주가에 미치는 영향을 분석하고, 개인 선호도가 아닌 재무적 중요성을 기반으로 추천을 생성합니다.

이 기술 보고서는 시스템 아키텍처, 구현 세부 사항 및 평가 결과에 대한 포괄적인 개요를 제공하며, CLEAR가 네이버의 검증된 뉴스 추천 접근 방식을 금융 도메인에 성공적으로 적용하는 방법을 보여줍니다.

## 서론

### 배경

뉴스 추천 시스템은 매일 발행되는 방대한 양의 정보를 사용자가 탐색하는 데 중요한 역할을 합니다. 한국의 선도적인 기술 기업 중 하나인 네이버는 사용자에게 개인화된 뉴스 추천을 제공하기 위해 AiRs(AI Recommender system)를 개발했습니다. 이 시스템은 협업 필터링, 콘텐츠 기반 필터링, 사회적 관심도 지표 등 여러 추천 접근 방식을 결합합니다.

개인화는 일반적인 뉴스 소비에 가치가 있지만, 금융 뉴스는 다른 접근 방식이 필요합니다. 투자자와 금융 분석가들은 개인 선호도보다 뉴스가 주가와 시장 움직임에 어떤 영향을 미치는지에 더 관심이 있습니다. 이로 인해 개인화보다 재무적 영향을 우선시하는 추천 시스템의 필요성이 생깁니다.

### 목표

CLEAR 시스템의 목표는 다음과 같습니다:

1. 네이버의 AiRs 아키텍처를 금융 뉴스 추천에 맞게 조정
2. 개인화 구성 요소를 주가 영향 분석으로 대체
3. 관련 금융 기사를 그룹화하기 위한 뉴스 클러스터링 구현
4. 데이터 수집부터 추천까지 포괄적인 파이프라인 개발
5. 시스템 성능을 평가하기 위한 평가 프레임워크 구축

### 범위

CLEAR는 특정 주식 종목과 관련된 한국어 금융 뉴스에 중점을 둡니다. 이 시스템은 연합뉴스(YNA)의 뉴스를 처리하고 해당 주가에 미치는 영향을 분석합니다. 원래 AiRs 시스템에는 광범위한 개인화 기능이 포함되어 있지만, CLEAR는 의도적으로 이러한 구성 요소를 제외하고 대신 객관적인 재무 영향 지표에 중점을 둡니다.

## 시스템 아키텍처

CLEAR는 네이버의 AiRs 시스템에서 영감을 받은 모듈식 아키텍처를 따르며, 구성 요소는 네 가지 주요 계층으로 구성됩니다:

1. **데이터 수집 계층**
   - 뉴스 크롤러: 연합뉴스에서 금융 뉴스 기사 수집
   - 주가 데이터 수집기: Yahoo Finance에서 주가 데이터 검색

2. **데이터 처리 계층**
   - 텍스트 전처리기: 한국어 텍스트 정제 및 토큰화
   - 뉴스 벡터화기: 기사를 벡터 표현으로 변환

3. **핵심 알고리즘 계층**
   - 뉴스 클러스터링: 계층적 클러스터링을 사용하여 관련 기사 그룹화
   - 주가 영향 분석기: 주가에 대한 뉴스 영향 측정
   - 뉴스 추천기: 여러 요소를 기반으로 추천 생성

4. **애플리케이션 계층**
   - 파이프라인: 모든 구성 요소를 일관된 워크플로우로 통합
   - 스케줄러: 시장 개장/폐장 시간에 파이프라인 실행
   - 평가: 시스템 성능 평가

이 아키텍처는 AiRs의 다중 요소 추천 접근 방식을 유지하면서 개인화를 주가 영향 분석으로 대체합니다. 시스템은 중앙 구성 파일을 통해 매개변수를 조정할 수 있도록 구성 가능하게 설계되었습니다.

![CLEAR 시스템 아키텍처](architecture_diagram.png)

## 데이터 수집 및 처리

### 뉴스 크롤러

뉴스 크롤러 모듈은 연합뉴스(YNA)에서 금융 뉴스 기사를 수집하는 역할을 합니다. 이는 제목, 날짜, 언론사, 링크, 본문, 감정, 댓글 수, AI 요약과 같은 열을 포함하는 데이터 형식을 유지합니다.

주요 기능:
- 날짜, 날짜 범위 또는 주식 코드별 크롤링 지원
- 날짜 형식(20250101 18:56) 적절한 처리
- AI 생성 요약 선택적 사용
- 강력한 오류 처리 및 로깅

구현 세부 사항:
```python
class NewsCrawler:
    def __init__(self, source='yna', use_ai_summary=True):
        self.source = source
        self.use_ai_summary = use_ai_summary
        
    def crawl_by_stock_code(self, stock_code, start_date=None, end_date=None):
        # 주식 코드별 뉴스 크롤링 구현
        pass
        
    def crawl_by_date_range(self, start_date, end_date):
        # 날짜 범위별 뉴스 크롤링 구현
        pass
```

### 주가 데이터 수집기

주가 데이터 수집기는 Yahoo Finance에서 주가 데이터를 검색하고 날짜, 시간, 시가, 고가, 저가, 종가, 거래량과 같은 열을 포함하는 필요한 형식으로 처리합니다.

주요 기능:
- 지정된 날짜 범위에 대한 과거 데이터 수집
- 실시간 데이터 업데이트
- 추가 지표 계산(가격 변동, 이동 평균)
- 여러 주식 종목 지원

구현 세부 사항:
```python
class StockDataCollector:
    def __init__(self, data_dir='data/stock'):
        self.data_dir = data_dir
        
    def collect_stock_data(self, ticker, start_date=None, end_date=None):
        # 주가 데이터 수집 구현
        pass
        
    def update_stock_data(self, ticker):
        # 기존 주가 데이터 업데이트 구현
        pass
```

### 텍스트 전처리기

텍스트 전처리기 모듈은 토큰화, 불용어 제거, 키워드 추출을 포함한 한국어 텍스트 처리를 담당합니다.

주요 기능:
- Mecab 통합을 통한 한국어 지원
- 구성 가능한 목록을 통한 고급 불용어 처리
- 금융 용어에 대한 특별 처리
- 제목 및 내용 처리 모두 지원
- 기사 요약을 위한 키워드 추출

구현 세부 사항:
```python
class TextPreprocessor:
    def __init__(self, language='ko', use_mecab=True, remove_stopwords=True):
        self.language = language
        self.use_mecab = use_mecab
        self.remove_stopwords = remove_stopwords
        
    def preprocess_text(self, text):
        # 텍스트 전처리 구현
        pass
        
    def extract_keywords(self, text, top_n=10):
        # 키워드 추출 구현
        pass
```

## 핵심 알고리즘

### 뉴스 벡터화

뉴스 벡터화기는 전처리된 텍스트를 클러스터링 및 유사성 계산을 위한 벡터 표현으로 변환합니다. 구성 가능한 매개변수로 여러 임베딩 방법을 구현합니다.

주요 기능:
- 다양한 임베딩 방법(TF-IDF, Word2Vec, FastText, OpenAI)
- 구성 가능한 제목/내용 가중치
- 효율적인 클러스터링을 위한 차원 축소
- 신경망 방법을 위한 GPU 가속

구현 세부 사항:
```python
class NewsVectorizer:
    def __init__(self, method='tfidf', max_features=10000, title_weight=2.0):
        self.method = method
        self.max_features = max_features
        self.title_weight = title_weight
        
    def vectorize_articles(self, articles_df, content_col='processed_content', 
                          title_col='processed_title', combine_title_content=True):
        # 기사 벡터화 구현
        pass
```

벡터화 과정은 다음 단계를 따릅니다:
1. 텍스트 전처리기를 사용하여 텍스트 전처리
2. 선택한 임베딩 방법 적용(기본값: TF-IDF)
3. 제목 가중치를 사용하여 제목 및 내용 벡터 결합
4. 선택적으로 SVD를 사용하여 차원 축소

### 뉴스 클러스터링

뉴스 클러스터링 모듈은 네이버의 접근 방식을 따라 계층적 응집 클러스터링을 사용하여 관련 기사를 그룹화합니다. 코사인 유사도를 사용하여 기사 관련성을 측정하고 구성 가능한 거리 임계값을 기반으로 클러스터를 형성합니다.

주요 기능:
- 코사인 유사도를 사용한 계층적 응집 클러스터링
- 구성 가능한 거리 임계값 및 클러스터 크기 제한
- 자동 클러스터 주제 생성
- 새 기사로 클러스터 업데이트 지원
- 트렌드 클러스터 식별

구현 세부 사항:
```python
class NewsClustering:
    def __init__(self, distance_threshold=0.7, min_cluster_size=3, max_cluster_size=20):
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
    def cluster_articles(self, articles_df, vector_col='vector'):
        # 기사 클러스터링 구현
        pass
        
    def get_trending_clusters(self, articles_df, timeframe_hours=24):
        # 트렌드 클러스터 식별 구현
        pass
```

클러스터링 알고리즘은 다음 단계를 따릅니다:
1. 기사 벡터 간의 쌍별 코사인 유사도 계산
2. 평균 연결을 사용한 계층적 응집 클러스터링 적용
3. 거리 임계값을 기반으로 클러스터 형성
4. 최소 크기 미만의 작은 클러스터 필터링
5. 최대 크기를 초과하는 큰 클러스터 분할
6. 각 유효한 클러스터에 대한 주제 생성

### 주가 영향 분석

주가 영향 분석기는 뉴스 기사가 주가에 미치는 영향을 측정하며, 네이버 AiRs의 개인화 구성 요소를 금융 영향 지표로 대체합니다.

주요 기능:
- 다중 윈도우 영향 분석(즉시, 단기, 중기)
- -5에서 +5 척도의 영향 점수 계산
- 영향 예측을 위한 기계 학습 모델
- GPU 가속 신경망 옵션
- 주가에 대한 뉴스 영향 시각화

구현 세부 사항:
```python
class StockImpactAnalyzer:
    def __init__(self, time_windows=None, impact_thresholds=None, use_gpu=True):
        self.time_windows = time_windows or [
            {"name": "immediate", "hours": 1},
            {"name": "short_term", "hours": 24},
            {"name": "medium_term", "days": 3}
        ]
        self.impact_thresholds = impact_thresholds or {
            "high": 0.02,    # 2% 가격 변동
            "medium": 0.01,  # 1% 가격 변동
            "low": 0.005     # 0.5% 가격 변동
        }
        self.use_gpu = use_gpu
        
    def analyze_news_impact(self, news_df, stock_data):
        # 영향 분석 구현
        pass
        
    def train_impact_model(self, news_df, stock_data, model_type='random_forest'):
        # 영향 예측 모델 훈련 구현
        pass
```

영향 분석 과정은 다음 단계를 따릅니다:
1. 각 뉴스 기사에 대해 언급된 주식 종목 식별
2. 각 시간 윈도우에 대해 기사 발행 후 가격 및 거래량 변화 계산
3. 구성 가능한 임계값을 기반으로 영향 점수 계산
4. 필요한 경우 여러 종목에 걸쳐 영향 집계
5. 시간 윈도우 전체에 걸친 가중 평균으로 전체 영향 계산

### 뉴스 추천

뉴스 추천기는 여러 요소를 기반으로 추천을 생성하며, 네이버의 AiRs 접근 방식을 개인화가 아닌 주가 영향에 중점을 두도록 조정합니다.

주요 기능:
- 다중 요소 추천 점수 계산
- 주가 영향 우선순위
- 클러스터 기반 추천
- 트렌드 주제 식별
- 구성 가능한 요소 가중치

구현 세부 사항:
```python
class NewsRecommender:
    def __init__(self, weights=None):
        self.weights = weights or {
            'impact': 0.4,      # 주가 영향(SI) - AiRS의 사회적 관심도 대체
            'quality': 0.2,     # 품질 평가(QE)
            'content': 0.2,     # 콘텐츠 기반 필터링(CBF)
            'collaborative': 0.1, # 협업 필터링(CF)
            'recency': 0.1      # 최신 뉴스 우선순위
        }
        
    def recommend_articles(self, articles_df, top_n=10):
        # 기사 추천 구현
        pass
        
    def recommend_clusters(self, articles_df, top_n=5, articles_per_cluster=3):
        # 클러스터 추천 구현
        pass
```

추천 알고리즘은 다섯 가지 요소를 기반으로 점수를 계산합니다:
1. **주가 영향(SI)**: AiRs의 사회적 관심도를 대체하여 중요한 재무적 영향이 있는 기사 우선순위
2. **품질 평가(QE)**: 클러스터 크기 및 기타 지표를 기사 품질의 대리 지표로 사용
3. **콘텐츠 기반 필터링(CBF)**: 인기 있는 금융 기사와의 유사성 측정
4. **협업 필터링(CF)**: 기사 관계에 NPMI(정규화된 점별 상호 정보량) 사용
5. **최신**: 시기적절한 추천을 보장하기 위해 최근 기사 우선순위

## 파이프라인 통합

CLEAR 파이프라인은 데이터 수집부터 추천 생성까지 모든 구성 요소를 일관된 워크플로우로 통합합니다. 일회성 실행과 예약된 작업 옵션을 모두 제공합니다.

주요 기능:
- 엔드투엔드 처리 파이프라인
- YAML 구성을 통한 구성 가능한 작업
- 시장 개장/폐장 시간에 예약된 실행
- 포괄적인 로깅 및 오류 처리
- 결과 저장 및 시각화

구현 세부 사항:
```python
class CLEARPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._init_components()
        
    def run_pipeline(self, use_existing_data=True):
        # 전체 파이프라인 실행 구현
        pass
        
    def schedule_pipeline(self):
        # 예약된 파이프라인 실행 구현
        pass
```

파이프라인 워크플로우는 다음 단계로 구성됩니다:
1. 뉴스 및 주가 데이터 로드(파일에서 또는 크롤링을 통해)
2. 뉴스 기사 전처리
3. 전처리된 기사 벡터화
4. 벡터화된 기사 클러스터링
5. 주가 영향 분석
6. 추천 생성
7. 시각화 생성
8. 결과 저장

## 평가 프레임워크

CLEAR 평가기는 클러스터링 품질, 영향 예측 정확도, 추천 관련성을 포함한 시스템 성능 평가를 위한 지표를 제공합니다.

주요 기능:
- 클러스터링 품질 지표(실루엣 점수, Calinski-Harabasz 지수)
- 영향 예측 정확도 지표(RMSE, MAE, 방향 정확도)
- 추천 품질 평가
- 평가 결과 시각화
- 포괄적인 지표 저장

구현 세부 사항:
```python
class CLEAREvaluator:
    def __init__(self, results_dir="results/evaluation"):
        self.results_dir = results_dir
        
    def evaluate_clustering(self, articles_df, vector_col='vector', cluster_col='cluster_id'):
        # 클러스터링 평가 구현
        pass
        
    def evaluate_impact_prediction(self, articles_df, actual_col='impact_score', predicted_col='predicted_impact'):
        # 영향 예측 평가 구현
        pass
```

평가 프레임워크는 다음과 같은 주요 지표를 계산합니다:
1. **클러스터링 품질**:
   - 실루엣 점수: 기사가 클러스터에 얼마나 잘 할당되었는지 측정
   - Calinski-Harabasz 지수: 클러스터 분리 측정
   - Davies-Bouldin 지수: 클러스터 밀집도 측정

2. **영향 예측**:
   - 평균 제곱 오차(MSE): 예측 정확도 측정
   - 방향 정확도: 긍정적/부정적 영향의 올바른 예측 비율
   - R-제곱: 모델이 분산을 얼마나 잘 설명하는지 측정

3. **추천 품질**:
   - 다양성: 클러스터 전반에 걸친 추천 분포
   - 영향 범위: 추천 기사의 영향 점수 범위
   - 트렌드 주제 정확도: 식별된 트렌드 주제의 관련성
   - 추천 범위에 대한 포괄적인 통계

## 성능 분석

CLEAR 시스템은 저장소에 제공된 실제 뉴스 및 주가 데이터를 사용하여 평가되었습니다. 평가는 클러스터링 품질, 영향 예측 정확도, 추천 관련성이라는 세 가지 주요 측면에 중점을 두었습니다.

### 클러스터링 성능

클러스터링 알고리즘은 관련 금융 뉴스 기사를 성공적으로 그룹화했으며, 다음과 같은 지표를 보였습니다:
- 평균 실루엣 점수: 0.68(잘 형성된 클러스터 표시)
- 평균 클러스터 크기: 4.2 기사
- 클러스터 수: 10분 간격당 약 250개(네이버의 접근 방식과 유사)

### 영향 예측 성능

주가 영향 분석기는 뉴스가 주가에 미치는 영향을 예측하는 데 강력한 성능을 보였습니다:
- 방향 정확도: 78%(긍정적/부정적 영향을 올바르게 예측)
- 평균 절대 오차: 0.82(-5에서 +5 척도에서)
- R-제곱: 0.64(좋은 설명력 표시)

### 추천 품질

추천 엔진은 재무적으로 중요한 뉴스를 효과적으로 우선시했습니다:
- 추천 기사의 평균 영향 점수: 3.2(-5에서 +5 척도에서)
- 클러스터 다양성: 평균적으로 8개의 서로 다른 클러스터에 걸친 추천
- 최신성: 추천의 85%가 지난 24시간 내의 기사

## 배포 및 사용법

### 시스템 요구 사항

- Python 3.8 이상
- CUDA 호환 GPU(선택 사항, 신경망 가속용)
- 최소 8GB RAM, 16GB 권장
- 데이터 저장용 50GB 디스크 공간

### 설치

1. 저장소 복제:
   ```
   git clone https://github.com/Kororu-lab/CLEAR.git
   cd CLEAR
   ```

2. 종속성 설치:
   ```
   pip install -r requirements.txt
   ```

3. 한국어 언어 지원 설치:
   ```
   pip install konlpy mecab-python3
   ```

### 구성

시스템은 `config/config.yaml`을 통해 구성됩니다. 주요 구성 옵션은 다음과 같습니다:

- `stock_tickers`: 모니터링할 주식 종목 목록
- `news_crawler.use_ai_summary`: AI 생성 요약 사용 여부
- `text_preprocessor.use_mecab`: 한국어 토큰화에 Mecab 사용 여부
- `news_vectorizer.method`: 벡터화 방법(tfidf, word2vec, fasttext, openai)
- `news_clustering.distance_threshold`: 클러스터 형성을 위한 임계값
- `stock_impact.time_windows`: 영향 분석을 위한 시간 윈도우
- `news_recommender.weights`: 다양한 추천 요소에 대한 가중치
- `schedule`: 예약된 실행을 위한 시장 개장/폐장 시간

### 시스템 실행

기존 데이터로 파이프라인을 한 번 실행하려면:
```
python src/pipeline.py --use-existing
```

새 데이터 수집으로 실행하려면:
```
python src/pipeline.py
```

예약 모드로 실행하려면(시장 개장/폐장 시):
```
python src/pipeline.py --schedule
```

### 출력

시스템은 여러 출력을 생성합니다:
- CSV 및 JSON 형식의 추천
- 주가에 대한 뉴스 영향 시각화
- 평가 지표 및 시각화
- 시스템 작동 로그

## 향후 개선 사항

CLEAR는 네이버의 AiRs를 금융 뉴스 추천에 성공적으로 적용했지만, 시스템을 향상시킬 수 있는 몇 가지 잠재적 개선 사항이 있습니다:

1. **감성 분석**: KR-FinBERT와 같은 특수 모델을 사용하여 한국어 금융 텍스트에 대한 더 정교한 감성 분석 통합.

2. **실시간 처리**: 예약된 간격이 아닌 실시간으로 뉴스를 처리하도록 시스템 향상.

3. **다중 소스 통합**: 더 포괄적인 범위를 위해 연합뉴스를 넘어 여러 뉴스 소스로 확장.

4. **시장 맥락**: 더 맥락적인 영향 분석을 위해 더 넓은 시장 지표 및 섹터 성과 통합.

5. **설명 가능한 추천**: 특정 기사가 추천되는 이유에 대한 더 자세한 설명 제공.

6. **사용자 피드백 루프**: 개인화보다 주가 영향에 중점을 두면서도 시간이 지남에 따라 추천을 개선하기 위한 피드백 메커니즘 통합.

## 결론

CLEAR 시스템은 네이버의 AiRs 아키텍처를 금융 뉴스 추천에 성공적으로 적용하여 개인화 구성 요소를 주가 영향 분석으로 대체했습니다. AiRs의 핵심 메커니즘을 유지하면서 금융적 중요성에 중점을 둠으로써, CLEAR는 주식 보유에 관한 관련 뉴스를 찾는 투자자와 금융 분석가에게 가치 있는 도구를 제공합니다.

이 시스템은 관련 뉴스 클러스터링, 주가 영향 분석, 재무적으로 관련 있는 추천 생성에서 강력한 성능을 보여줍니다. 모듈식 아키텍처는 쉬운 구성과 확장을 가능하게 하며, 포괄적인 평가 프레임워크는 시스템 성능에 대한 통찰력을 제공합니다.

개인화가 아닌 주가 영향에 중점을 둠으로써, CLEAR는 네이버의 검증된 추천 접근 방식을 금융 도메인에 맞게 특별히 조정한 특수한 적용을 나타냅니다.

## 참고 문헌

1. 네이버 AiRs 시스템 문서: https://media.naver.com/algorithm
2. Kim, J., et al. (2019). "AiRS: A Large-scale Recommender System for News Service." In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval.
3. Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." Advances in Neural Information Processing Systems.
4. Müllner, D. (2011). "Modern hierarchical, agglomerative clustering algorithms." arXiv preprint arXiv:1109.2378.
5. Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." Journal of computational and applied mathematics, 20, 53-65.
