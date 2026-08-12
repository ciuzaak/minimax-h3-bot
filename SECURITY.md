# Security

Please do not open a public issue for a suspected credential leak or another
security-sensitive problem. Use GitHub's private vulnerability reporting for
this repository when available, or contact the repository owner privately.

Never commit `.env`, Telegram bot tokens, user allowlists, generated media, or
machine-specific runtime data. If a credential is committed, revoke it first;
removing it from the latest revision is not sufficient because Git retains
earlier versions.

