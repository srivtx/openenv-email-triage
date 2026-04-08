# 05 - Deploy and Submit (Exact Practical Flow)

This chapter is the shortest path from local project to accepted submission checks.

## 1. Local Readiness Checklist

Run in repo root:

```bash
python -m pytest -q
openenv validate
python inference.py
```

Expected:

- tests pass
- OpenEnv validate passes
- inference logs print START/STEP/END

## 2. Push to GitHub

If new repo:

```bash
git init
git add .
git commit -m "Initial OpenEnv environment"
git branch -M main
git remote add origin <your_repo_url>
git push -u origin main
```

## 3. Create Hugging Face Token (Write)

- Go to HF settings -> Access Tokens
- Create new token with **Write** role
- Export it in shell:

```bash
export HF_TOKEN="hf_xxx"
```

## 4. Create/Push Space

Use OpenEnv push:

```bash
openenv push -r <hf_username>/<space_name> --no-interface
```

Why `--no-interface`?

- avoids auto-web-ui metadata quirks
- API-only deployment is enough for OpenEnv checks

## 5. Verify Deployed Endpoints

```bash
curl -s https://<space-subdomain>.hf.space/health
curl -s -X POST https://<space-subdomain>.hf.space/reset -H 'Content-Type: application/json' -d '{}'
```

Expected HTTP 200 + valid JSON.

## 6. Submission URLs

Provide exactly:

- GitHub repo URL
- Hugging Face Space URL

If asked for API base URL too, use:

- `https://<space-subdomain>.hf.space`

## 7. What "Phase 1 passed" Means

It means technical gate checks passed:

- reset endpoint works
- Dockerfile location accepted
- inference.py detected
- openenv validate accepted

Then Phase 2 model scoring starts.

## 8. Security Cleanup (Important)

If token was ever shared in chat/logs:

- revoke token
- create a fresh token
- update local shell secret
