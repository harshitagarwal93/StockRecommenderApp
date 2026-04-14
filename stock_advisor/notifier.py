"""Email notification for daily recommendations (optional)."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .config import Config
from .models import DailyRecommendation

logger = logging.getLogger(__name__)


def send_recommendation_email(config: Config, rec: DailyRecommendation) -> bool:
    if not config.smtp_user or not config.notification_email:
        logger.info("Email not configured — skipping notification")
        return False

    subject = f"Stock Advisor — Daily Recommendation ({rec.date})"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.smtp_user
    msg["To"] = config.notification_email
    msg.attach(MIMEText(_build_plaintext(rec), "plain"))

    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
            server.starttls()
            server.login(config.smtp_user, config.smtp_password)
            server.sendmail(config.smtp_user, config.notification_email, msg.as_string())
        logger.info("Email sent to %s", config.notification_email)
        return True
    except Exception:
        logger.exception("Failed to send email")
        return False


def _build_plaintext(rec: DailyRecommendation) -> str:
    lines = [
        f"Stock Advisor — {rec.date}",
        f"Portfolio: Rs.{rec.portfolio_value:,.0f} ({rec.portfolio_returns_pct:+.1f}%)",
        f"Outlook: {rec.market_outlook}",
        "",
    ]
    for r in rec.recommendations:
        lines.append(
            f"  {r['action']} {r.get('quantity', 0)} x {r['ticker']} @ Rs.{r.get('current_price', 0):,.2f}"
        )
        lines.append(f"    Reason: {r.get('reason', '')}")
    if not rec.recommendations:
        lines.append("  No action recommended today.")
    return "\n".join(lines)
