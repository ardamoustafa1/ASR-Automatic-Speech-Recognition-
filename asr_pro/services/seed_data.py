"""Default topics, keyword rules, and alert rules."""

import os

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from asr_pro.db.models import AlertRule, KeywordRule, Topic, User, new_uuid

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DEFAULT_TOPICS = [
    {
        "slug": "cancellation",
        "label_tr": "İptal",
        "seed_keywords": ["iptal", "abonelik iptali", "hattı kapat", "sözleşme iptali"],
        "synonyms": ["iptal etmek", "vazgeçmek"],
    },
    {
        "slug": "complaint",
        "label_tr": "Şikayet",
        "seed_keywords": ["şikayet", "sikayet", "memnun değilim", "kötü hizmet"],
        "synonyms": ["şikayetim var", "memnuniyetsizlik"],
    },
    {
        "slug": "competitor",
        "label_tr": "Rakip Firma",
        "seed_keywords": ["vodafone", "turkcell", "türk telekom", "turk telekom"],
        "synonyms": ["rakip operatör", "başka firma"],
    },
    {
        "slug": "price_increase",
        "label_tr": "Zam",
        "seed_keywords": ["zam", "fiyat artışı", "zamlı", "tarife yükseltme"],
        "synonyms": ["ücret artışı", "fiyat zammı"],
    },
    {
        "slug": "invoice",
        "label_tr": "Fatura",
        "seed_keywords": ["fatura", "fatura itirazı", "yanlış yansıma", "fatura hatası"],
        "synonyms": ["fatura tutarı", "fatura bedeli"],
    },
    {
        "slug": "refund",
        "label_tr": "İade",
        "seed_keywords": ["iade", "para iadesi", "geri ödeme"],
        "synonyms": ["iade talebi", "ücret iadesi"],
    },
]


DEFAULT_RULES = [
    {
        "name": "İptal Tespiti",
        "slug": "cancellation",
        "keywords": ["iptal", "abonelik iptali", "hattı kapat"],
        "severity": "warning",
        "match_mode": "semantic",
    },
    {
        "name": "Şikayet Tespiti",
        "slug": "complaint",
        "keywords": ["şikayet", "sikayet", "memnun değilim"],
        "severity": "critical",
        "match_mode": "semantic",
    },
    {
        "name": "Rakip Firma",
        "slug": "competitor",
        "keywords": ["vodafone", "turkcell", "türk telekom"],
        "severity": "info",
        "match_mode": "exact",
    },
    {
        "name": "Zam Tespiti",
        "slug": "price_increase",
        "keywords": ["zam"],
        "severity": "warning",
        "match_mode": "semantic",
    },
    {
        "name": "Fatura Tespiti",
        "slug": "invoice",
        "keywords": ["fatura", "fatura itirazı", "yanlış yansıma"],
        "severity": "info",
        "match_mode": "semantic",
    },
    {
        "name": "İade Tespiti",
        "slug": "refund",
        "keywords": ["iade", "para iadesi", "geri ödeme"],
        "severity": "info",
        "match_mode": "semantic",
    },
]


def seed_defaults(db: Session) -> None:
    existing_topics = {t.slug: t for t in db.query(Topic).all()}
    slug_to_id: dict[str, str] = {}

    for item in DEFAULT_TOPICS:
        if item["slug"] in existing_topics:
            slug_to_id[item["slug"]] = existing_topics[item["slug"]].id
            continue
        topic = Topic(
            id=new_uuid(),
            slug=item["slug"],
            label_tr=item["label_tr"],
            seed_keywords=item["seed_keywords"],
            synonyms=item.get("synonyms", []),
        )
        db.add(topic)
        slug_to_id[item["slug"]] = topic.id

    db.flush()

    existing_rules = db.query(KeywordRule).count()
    if existing_rules == 0:
        for item in DEFAULT_RULES:
            rule = KeywordRule(
                id=new_uuid(),
                name=item["name"],
                keywords=item["keywords"],
                match_mode=item.get("match_mode", "semantic"),
                severity=item["severity"],
                topic_id=slug_to_id.get(item["slug"]),
                is_active=True,
            )
            db.add(rule)

    existing_alerts = db.query(AlertRule).count()
    if existing_alerts == 0:
        zam_rule = db.query(KeywordRule).filter(KeywordRule.name == "Zam Tespiti").first()
        if zam_rule:
            alert = AlertRule(
                id=new_uuid(),
                name="Zam %40 Artış Uyarısı",
                target_type="keyword",
                target_id=zam_rule.id,
                condition={
                    "metric": "pct_increase",
                    "operator": "pct_increase",
                    "threshold": 40,
                    "window": "7d",
                    "compare_to": "prev_7d",
                    "min_count": 5,
                },
                channels=["in_app"],
                cooldown_minutes=1440,
                is_active=True,
            )
            db.add(alert)

    import secrets

    from loguru import logger

    def _seed_user(username: str, env_var: str, role: str, team: str | None = None) -> None:
        if db.query(User).filter(User.username == username).first():
            return
        password = os.environ.get(env_var)
        if not password:
            if os.getenv("ASR_ENV") == "prod":
                raise RuntimeError(f"{env_var} must be set in production.")
            password = secrets.token_urlsafe(16)
            logger.warning(
                f"{env_var} not set. Generated random password for '{username}': {password}"
            )
        db.add(
            User(
                id=new_uuid(),
                username=username,
                hashed_password=pwd_context.hash(password),
                role=role,
                team=team,
                is_active=True,
            )
        )

    # Demo users covering every role in the RBAC matrix. `agent` and `team_lead`
    # share a team so team-scoped conversation visibility is observable out of the box.
    _seed_user("admin", "ASR_ADMIN_PASSWORD", "admin")
    _seed_user("agent", "ASR_AGENT_PASSWORD", "agent", team="team_alpha")
    _seed_user("team_lead", "ASR_TEAM_LEAD_PASSWORD", "team_lead", team="team_alpha")
    _seed_user("qa", "ASR_QA_PASSWORD", "qa")
    _seed_user("auditor", "ASR_AUDITOR_PASSWORD", "auditor")

    db.commit()
