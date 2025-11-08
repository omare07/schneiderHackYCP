# Git History Cleanup Instructions - API Key Exposure

## ⚠️ CRITICAL SECURITY ISSUE

API keys were exposed in previous commits. This document provides step-by-step instructions to remove them from Git history.

## Exposed API Keys

The following API key was found in Git history:
- `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`

**Files that contained hardcoded keys:**
- `spectral-analyzer-web/backend/config/env_config.py`
- `spectral-analyzer-web/backend/api/routes/analysis.py`
- `spectral-analyzer-web/backend/.env`
- `spectral_analyzer/live_test.py`
- `spectral_analyzer/standalone_test.py`

## Option 1: Using git-filter-repo (Recommended)

`git-filter-repo` is the modern, recommended tool for rewriting Git history.

### Installation

```bash
# macOS
brew install git-filter-repo

# Linux (pip)
pip3 install git-filter-repo

# Or download directly
curl -O https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
chmod +x git-filter-repo
```

### Steps to Clean History

1. **Backup your repository first:**
   ```bash
   cd /Users/omarelfernani/Downloads/schneiderHackYCP
   cd ..
   cp -r schneiderHackYCP schneiderHackYCP-backup
   cd schneiderHackYCP
   ```

2. **Create a file with sensitive strings to remove:**
   ```bash
   cat > /tmp/api-keys-to-remove.txt << 'EOF'
   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
   EOF
   ```

3. **Run git-filter-repo to remove the API keys:**
   ```bash
   git-filter-repo --replace-text /tmp/api-keys-to-remove.txt --force
   ```

4. **Verify the changes:**
   ```bash
   # Search for any remaining instances of the old key
   git log --all --source -S "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"
   
   # Should return no results
   ```

5. **Force push to remote (CRITICAL - This rewrites history):**
   ```bash
   git remote add origin <your-github-url>  # if not already added
   git push origin --force --all
   git push origin --force --tags
   ```

## Option 2: Using BFG Repo-Cleaner (Alternative)

BFG is faster and simpler for large repositories.

### Installation

```bash
# Download BFG
curl -L https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -o ~/bfg.jar
```

### Steps to Clean History

1. **Backup your repository:**
   ```bash
   cd /Users/omarelfernani/Downloads
   cp -r schneiderHackYCP schneiderHackYCP-backup
   ```

2. **Create a replacement file:**
   ```bash
   cat > ~/api-keys.txt << 'EOF'
   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==>***REMOVED***
   EOF
   ```

3. **Run BFG:**
   ```bash
   cd /Users/omarelfernani/Downloads/schneiderHackYCP
   java -jar ~/bfg.jar --replace-text ~/api-keys.txt
   ```

4. **Clean up and push:**
   ```bash
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   git push origin --force --all
   git push origin --force --tags
   ```

## Option 3: Delete and Recreate Repository (Simplest)

If this is a new repository with minimal history:

1. **Backup important files**
2. **Delete the GitHub repository**
3. **Remove local .git directory:**
   ```bash
   cd /Users/omarelfernani/Downloads/schneiderHackYCP
   rm -rf .git
   ```
4. **Initialize fresh repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - clean repository with environment variables"
   ```
5. **Create new GitHub repository and push:**
   ```bash
   git remote add origin <new-github-url>
   git branch -M main
   git push -u origin main
   ```

## Post-Cleanup Verification

After cleaning Git history, verify the cleanup was successful:

```bash
# Clone the repository fresh to verify
cd /tmp
git clone <your-github-url> test-clone
cd test-clone

# Search for the old API key
grep -r "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" .
git log --all -S "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"

# Should return no results
```

## Important Security Actions

### 1. Rotate API Key with OpenRouter

Even after cleaning Git history, you should rotate the exposed key:

1. Visit https://openrouter.ai/keys
2. Revoke the old key: `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`
3. Generate a new API key
4. Update your `.env` files with the new key

### 2. Notify Team Members

If working with a team:
- Inform all team members that Git history was rewritten
- They will need to re-clone the repository or reset their local copies
- Share the new API key securely (not via email/chat)

### 3. Update Local Clones

For team members with existing clones:

```bash
cd /path/to/repository
git fetch origin
git reset --hard origin/main  # or your default branch
```

## Prevention for Future

The following changes have been implemented to prevent future exposures:

✅ All hardcoded API keys replaced with environment variables
✅ `.env` files added to `.gitignore`
✅ `.env.example` files created as templates
✅ Environment variable validation added to code
✅ Pre-commit hooks recommended (see below)

### Recommended: Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml (already done)
# Run pre-commit install
pre-commit install
```

## Support & Resources

- git-filter-repo: https://github.com/newren/git-filter-repo
- BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/
- GitHub: Removing sensitive data: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
- OpenRouter Security: https://openrouter.ai/docs/security

## Questions?

Contact your team lead or security officer for assistance.