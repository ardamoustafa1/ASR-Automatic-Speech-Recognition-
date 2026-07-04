# Aggregates call center analytics to detect emerging topic and sentiment trends.
"""Top-Tier Customer Journey & Early Warning Trend Engine (SQLAlchemy refactored)."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func

from asr_pro.db.models import TrendCallLog
from asr_pro.db.session import init_db as _core_init_db
from asr_pro.db.session import session_scope


@dataclass
class AnomalyAlert:
    topic: str
    baseline_avg: float
    recent_count: int
    increase_percentage: int
    severity: str  # "CRITICAL", "WARNING"


@dataclass
class ForecastResult:
    topic: str
    predicted_volume: int
    trend_slope: float
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"


def init_db():
    _core_init_db()
    with session_scope():
        # Sadece gerçek veri kullanılacak, mock data üretimi iptal edildi.
        pass


def log_call_trend(topic: str):
    """Yeni biten bir çağrının konusunu hafızaya yazar."""
    with session_scope() as session:
        log_entry = TrendCallLog(topic=topic, call_date=datetime.now())
        session.add(log_entry)


def get_trend_data(days: int = 14) -> dict[str, dict[str, int]]:
    """Son X günün topic tabanlı şikayet hacmini getirir. (Streamlit line_chart formatı için idealdir)"""
    init_db()

    start_date = datetime.now().date() - timedelta(days=days - 1)
    dt_start = datetime.combine(start_date, datetime.min.time())

    with session_scope() as session:
        # SQLite'da date() fonksiyonunu emule etmek zordur (Postgres vs değişir).
        # En taşınabilir yöntem veriyi çekip Python'da gruplamaktır,
        # ancak hacim büyükse SQLAlchemy func ile gruplanmalıdır.

        logs = session.query(TrendCallLog).filter(TrendCallLog.call_date >= dt_start).all()

        all_topics = session.query(TrendCallLog.topic).distinct().all()
        all_topics = [t[0] for t in all_topics]

        # Format: { '2023-10-01': {'Mobil': 15, 'Ödeme': 20}, '2023-10-02': {...} }
        data_by_date = {}

        # Öncelikle tüm tarihleri ve tüm topicleri 0 ile başlat (eksik günleri doldur)
        for i in range(days):
            d_str = (datetime.now().date() - timedelta(days=days - 1 - i)).isoformat()
            data_by_date[d_str] = dict.fromkeys(all_topics, 0)

        for log in logs:
            d_str = (
                log.call_date.date().isoformat()
                if hasattr(log.call_date, "date")
                else str(log.call_date)[:10]
            )
            if d_str in data_by_date:
                data_by_date[d_str][log.topic] = data_by_date[d_str].get(log.topic, 0) + 1

        return data_by_date


def detect_anomalies(data_by_date: dict[str, dict[str, int]]) -> list[AnomalyAlert]:
    """Z-Score ve Percentage Spike mantığıyla son 1-2 gündeki patlamaları bulur (Early Warning)."""
    if not data_by_date:
        return []

    dates = sorted(data_by_date.keys())
    if len(dates) < 3:
        return []  # Yeterli geçmiş yok

    history_dates = dates[:-2]  # Son 2 gün hariç geçmiş
    recent_dates = dates[-2:]  # Son 2 gün

    if not history_dates:
        return []

    topics = list(data_by_date[dates[0]].keys())
    alerts = []

    for topic in topics:
        # Geçmiş ortalama hesapla
        history_counts = [data_by_date[d].get(topic, 0) for d in history_dates]
        baseline_avg = sum(history_counts) / len(history_counts) if history_counts else 0.0

        # Son günlerin ortalaması
        recent_counts = [data_by_date[d].get(topic, 0) for d in recent_dates]
        recent_avg = sum(recent_counts) / len(recent_counts) if recent_counts else 0.0

        # Eğer baseline çok düşükse küçük artışlarda sahte alarm vermesin diye threshold koyalım
        if baseline_avg < 5:
            baseline_avg = 5.0

        increase_pct = ((recent_avg - baseline_avg) / baseline_avg) * 100

        if increase_pct >= 100:
            alerts.append(
                AnomalyAlert(
                    topic=topic,
                    baseline_avg=round(baseline_avg, 1),
                    recent_count=int(recent_avg),
                    increase_percentage=int(increase_pct),
                    severity="CRITICAL",
                )
            )
        elif increase_pct >= 50:
            alerts.append(
                AnomalyAlert(
                    topic=topic,
                    baseline_avg=round(baseline_avg, 1),
                    recent_count=int(recent_avg),
                    increase_percentage=int(increase_pct),
                    severity="WARNING",
                )
            )

    # En yüksek artışa göre sırala
    return sorted(alerts, key=lambda x: x.increase_percentage, reverse=True)


def forecast_tomorrow(
    data_by_date: dict[str, dict[str, int]], days_to_look_back: int = 7
) -> list[ForecastResult]:
    """
    Doğrusal Regresyon (Linear Regression) ile yarının çağrı hacmini tahmin eder.
    Sadece belirgin bir artış trendi olan (Slope > 0) konuları döndürür.
    """
    if not data_by_date:
        return []

    dates = sorted(data_by_date.keys())
    if len(dates) < days_to_look_back:
        return []

    recent_dates = dates[-days_to_look_back:]
    topics = list(data_by_date[dates[0]].keys())

    forecasts = []

    # x ekseni (Gün indeksi 0, 1, 2... N-1)
    N = days_to_look_back
    x = list(range(N))
    sum_x = sum(x)
    sum_x_sq = sum([i**2 for i in x])

    for topic in topics:
        y = [data_by_date[d].get(topic, 0) for d in recent_dates]
        sum_y = sum(y)
        sum_xy = sum([x[i] * y[i] for i in range(N)])

        # Eğim (Slope / m) formülü
        denominator = N * sum_x_sq - sum_x**2
        if denominator == 0:
            continue

        m = (N * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y - m * sum_x) / N

        # Yarının tahmini (x = N)
        predicted = m * N + b

        # Yalnızca artış trendinde olanları ve anlamlı hacme sahip olanları tahminle
        if m > 0.5 and predicted > 5:
            confidence = "HIGH" if m > 2.0 else "MEDIUM"
            forecasts.append(
                ForecastResult(
                    topic=topic,
                    predicted_volume=int(predicted),
                    trend_slope=round(m, 2),
                    confidence_level=confidence,
                )
            )

            # En keskin artışa göre sırala
    return sorted(forecasts, key=lambda f: f.trend_slope, reverse=True)


@dataclass
class TrendResult:
    keyword: str
    current_count: int
    previous_count: int
    pct_change: float


def compute_trend(db, window="7d", topic_id=None, rule_id=None) -> TrendResult:
    """
    Computes the percentage change of keyword hits over a given time window.

    Args:
        db (Session): SQLAlchemy database session.
        window (str, optional): Time window to calculate trend (e.g. "7d", "14d"). Defaults to "7d".
        topic_id (str, optional): Filter by topic ID. Defaults to None.
        rule_id (str, optional): Filter by rule ID. Defaults to None.

    Returns:
        TrendResult: Dataclass containing current count, previous count, and percentage change.
    """
    from datetime import datetime, timedelta, timezone

    from asr_pro.db.models import KeywordHit

    days = int(window.replace("d", "")) if "d" in window else 7
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(days=days)
    prev_start = current_start - timedelta(days=days)

    query = db.query(KeywordHit)
    if topic_id:
        query = query.filter(KeywordHit.topic_id == topic_id)
    if rule_id:
        query = query.filter(KeywordHit.rule_id == rule_id)

    current_count = query.filter(KeywordHit.created_at >= current_start).count()
    previous_count = query.filter(
        KeywordHit.created_at >= prev_start, KeywordHit.created_at < current_start
    ).count()

    pct_change = 0.0
    if previous_count > 0:
        pct_change = ((current_count - previous_count) / previous_count) * 100.0
    elif current_count > 0:
        pct_change = 100.0

    keyword_name = "Belirtilen Kural/Konu"
    return TrendResult(
        keyword=keyword_name,
        current_count=current_count,
        previous_count=previous_count,
        pct_change=round(pct_change, 2),
    )


def top_keywords(db, limit=5, window="7d"):
    """
    Retrieves the most frequently hit keywords in the specified time window.

    Args:
        db (Session): SQLAlchemy database session.
        limit (int, optional): Number of top keywords to return. Defaults to 5.
        window (str, optional): Time window (e.g. "7d"). Defaults to "7d".

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing 'keyword' and 'count'.
    """
    from datetime import datetime, timedelta, timezone

    from asr_pro.db.models import KeywordHit

    days = int(window.replace("d", "")) if "d" in window else 7
    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    results = (
        db.query(KeywordHit.keyword, func.count(KeywordHit.id).label("hit_count"))
        .filter(KeywordHit.created_at >= start_date)
        .group_by(KeywordHit.keyword)
        .order_by(func.count(KeywordHit.id).desc())
        .limit(limit)
        .all()
    )

    return [{"keyword": r[0], "count": r[1]} for r in results]


def dashboard_summary(db, window="7d"):
    """
    Provides a high-level summary of total calls, average risk, and active alerts.

    Args:
        db (Session): SQLAlchemy database session.
        window (str, optional): Time window to summarize (e.g. "7d"). Defaults to "7d".

    Returns:
        Dict[str, Any]: Dictionary with 'total_calls', 'avg_risk', and 'active_alerts'.
    """
    from datetime import datetime, timedelta, timezone

    from asr_pro.db.models import AlertEvent, Conversation

    days = int(window.replace("d", "")) if "d" in window else 7
    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    conversations = db.query(Conversation).filter(Conversation.created_at >= start_date).all()
    total_calls = len(conversations)

    active_alerts = (
        db.query(AlertEvent)
        .filter(AlertEvent.acknowledged is False, AlertEvent.created_at >= start_date)
        .count()
    )

    avg_risk = 0.0
    if total_calls > 0:
        total_risk = sum(
            c.metadata_json.get("churn_risk", 0.0) for c in conversations if c.metadata_json
        )
        avg_risk = total_risk / total_calls

    # Also calculate hit counts today
    from asr_pro.db.models import KeywordHit

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hits_today = db.query(KeywordHit).filter(KeywordHit.created_at >= today_start).count()

    # And get the top rising keyword to display
    from asr_pro.core.trend_engine import detect_anomalies, get_trend_data

    alerts = detect_anomalies(get_trend_data(days))
    top_rising = None
    if alerts:
        top_rising = {"keyword": alerts[0].topic, "increase": alerts[0].increase_percentage}

    return {
        "conversations_total": total_calls,
        "avg_risk": round(avg_risk, 2),
        "active_alerts": active_alerts,
        "hits_today": hits_today,
        "top_rising_keyword": top_rising,
    }
