# Security Recommendations & Best Practices

## 🚨 IMMEDIATE ACTIONS REQUIRED

### 1. Clean Git History (CRITICAL!)
The exposed API key `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` still exists in Git history.

**Action:** Follow [`GIT_HISTORY_CLEANUP_INSTRUCTIONS.md`](GIT_HISTORY_CLEANUP_INSTRUCTIONS.md) immediately.

### 2. Rotate API Key (CRITICAL!)
Even after cleaning Git history, the exposed key should be rotated.

**Steps:**
1. Visit https://openrouter.ai/keys
2. Revoke key: `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`
3. Generate a new API key
4. Update all `.env` files with the new key
5. Restart all running services

### 3. Verify Repository Security
```bash
# Clone fresh copy and search for exposed keys
cd /tmp
git clone <your-repo-url> security-check
cd security-check
grep -r "sk-or-v1-" .
git log --all -S "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"
```

## ✅ Completed Security Fixes

### Code Changes
- ✅ All hardcoded API keys replaced with `os.getenv('OPENROUTER_API_KEY')`
- ✅ Environment variable validation added (raises error if missing)
- ✅ Test files updated to use environment variables
- ✅ Import of `python-dotenv` added where needed

### Configuration Files
- ✅ `.env` files updated with new API key
- ✅ `.env.example` files created as templates
- ✅ `.gitignore` enhanced to exclude all `.env` variants

### Files Modified
1. [`spectral-analyzer-web/backend/api/routes/analysis.py`](spectral-analyzer-web/backend/api/routes/analysis.py:399)
2. [`spectral-analyzer-web/backend/config/env_config.py`](spectral-analyzer-web/backend/config/env_config.py:14)
3. [`spectral_analyzer/live_test.py`](spectral_analyzer/live_test.py)
4. [`spectral_analyzer/standalone_test.py`](spectral_analyzer/standalone_test.py)
5. [`.gitignore`](.gitignore:45-51)

## 📋 Long-term Security Best Practices

### 1. Environment Variable Management

**Never commit `.env` files:**
```bash
# Verify .env files are ignored
git status
# .env files should NOT appear in untracked files
```

**Always use `.env.example` templates:**
- Keep `.env.example` in version control
- Document all required environment variables
- Use placeholder values (e.g., `your-api-key-here`)

### 2. Code Review Checklist

Before committing, always check:
```bash
# Search for potential API keys
grep -r "sk-" . --exclude-dir=node_modules --exclude-dir=.git
grep -r "api[_-]key\s*=\s*['\"]" . --exclude-dir=node_modules --exclude-dir=.git

# Review what you're committing
git diff --cached
```

### 3. Pre-commit Hooks (Recommended)

Install pre-commit hooks to catch secrets before they're committed:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: detect-private-key
      
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
EOF

# Install hooks
pre-commit install
```

### 4. Secret Scanning with GitHub

Enable GitHub's secret scanning:
1. Go to repository settings
2. Security & analysis
3. Enable "Secret scanning"
4. Enable "Push protection" (prevents commits with secrets)

### 5. API Key Rotation Schedule

Implement regular API key rotation:
- **Production keys:** Rotate every 90 days
- **Development keys:** Rotate every 180 days
- **After any security incident:** Immediately

### 6. Access Control

**Limit who has access to production API keys:**
- Use separate keys for dev/staging/production
- Implement principle of least privilege
- Use a secrets manager (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault)

### 7. Monitoring & Alerting

**Set up monitoring for:**
- Unusual API usage patterns
- API calls from unexpected IPs
- Rate limit violations
- Failed authentication attempts

**OpenRouter specific:**
- Monitor usage at https://openrouter.ai/usage
- Set up billing alerts
- Review API access logs regularly

### 8. Documentation

**Always document:**
- Which environment variables are required
- How to obtain API keys
- Where to report security issues
- Key rotation procedures

### 9. Incident Response Plan

**If an API key is exposed:**
1. Immediately revoke the key
2. Generate a new key
3. Update all environments
4. Review access logs for unauthorized usage
5. Clean Git history (if committed)
6. Document the incident
7. Implement additional controls to prevent recurrence

## 🛡️ Additional Security Measures

### Use Secret Management Services

**Recommended tools:**
- **AWS Secrets Manager** - For AWS deployments
- **Azure Key Vault** - For Azure deployments
- **HashiCorp Vault** - For multi-cloud or on-premise
- **Doppler** - Developer-friendly secrets management
- **1Password CLI** - For small teams

Example with AWS Secrets Manager:
```python
import boto3
import json

def get_api_key():
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )
    
    secret = client.get_secret_value(SecretId='openrouter/api-key')
    return json.loads(secret['SecretString'])['api_key']
```

### Rate Limiting & Quotas

Implement application-level rate limiting:
```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=100, period=60)  # 100 calls per minute
def call_openrouter_api():
    # Your API call here
    pass
```

### IP Whitelisting

If OpenRouter supports it:
- Restrict API key usage to specific IP addresses
- Use VPN or static IPs for production environments

### Logging Best Practices

**Never log sensitive data:**
```python
# ❌ BAD
logger.info(f"API key: {api_key}")

# ✅ GOOD
logger.info("API key loaded successfully")
logger.info(f"API key prefix: {api_key[:10]}...")
```

## 📞 Contact & Support

**Report security issues:**
- Email: security@yourcompany.com
- Private vulnerability disclosure: https://github.com/yourorg/repo/security

**Resources:**
- OpenRouter Security: https://openrouter.ai/docs/security
- OWASP API Security: https://owasp.org/www-project-api-security/
- GitHub Secret Scanning: https://docs.github.com/en/code-security/secret-scanning

## 🔄 Regular Security Audits

**Schedule regular reviews:**
- **Weekly:** Manual code review for secrets
- **Monthly:** Review API key usage and access logs
- **Quarterly:** Full security audit and key rotation
- **Annually:** Third-party security assessment

## ✅ Security Checklist

Use this checklist for every deployment:

- [ ] No hardcoded API keys or secrets
- [ ] All `.env` files in `.gitignore`
- [ ] `.env.example` is up to date
- [ ] Environment variables validated at startup
- [ ] Pre-commit hooks installed
- [ ] API keys rotated if needed
- [ ] Access logs reviewed
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team notified of changes

---

**Last Updated:** 2025-11-08
**Next Review:** 2026-02-08