# Wiz Exercise — Session Handoff Context

## Who I am
Oleksii Androsov, Principal Solutions Architect, Frankfurt. NOT a software engineer —
explain every non-obvious line of code. I learn by building. Deep AWS background, new to
Docker/Kubernetes/MongoDB. Full background in CLAUDE.md at repo root.

## What this is
Closing round interview exercise for Wiz (cloud security company). I need to build a
deliberately insecure cloud environment, then present it with security findings.
Interview format: 60 min — 15 min presentation, 25 min Q&A on what I built, 15 min Wiz discussion.
Deadline: ~1 week from June 10, 2026.

## The application
MovieBuddy — a Streamlit-based AI movie recommendation chatbot. Multi-agent system using
Anthropic Claude API. Previously deployed on EC2, accessible at movie-buddy.app.
Code is in: wiz-exercise/app/

## What needs to be built (full architecture)

### Infrastructure
- EC2 VM (intentionally outdated Ubuntu 20.04 + MongoDB 4.4, SSH open to internet,
  overly permissive IAM role) — these weaknesses are ON PURPOSE for the Wiz demo
- EKS cluster (Kubernetes) in private subnet
- S3 bucket for MongoDB backups (public read — intentional weakness)
- ECR repository for Docker images
- AWS Load Balancer exposing the app publicly
- DNS: movie-buddy.app will point to the new Load Balancer

### Application changes needed
1. memory.py — replace all DynamoDB/boto3 code with MongoDB/pymongo
   (profiles, summaries, devices tables → MongoDB collections)
2. requirements.txt — add pymongo, streamlit; remove boto3
3. Dockerfile — already written at wiz-exercise/app/Dockerfile
4. wizexercise.txt — already created with "Oleksii Androsov" ✓

### Kubernetes config (wiz-exercise/k8s/)
- deployment.yaml — runs MovieBuddy container, MongoDB URL as env var
- clusterrolebinding.yaml — cluster-admin role on pod (intentional weakness)
- ingress.yaml — exposes via AWS ALB

### Two GitHub Actions pipelines (wiz-exercise/.github/workflows/)
- Pipeline 1: push to infra/ → terraform apply (deploy infrastructure)
- Pipeline 2: push to app/ → docker build → push to ECR → kubectl rollout

### Pipeline security controls
- Checkov: scans Terraform for misconfigs before deploy
- Trivy: scans Docker image for CVEs before push
- Branch protection on GitHub main branch

### Cloud Native Security (AWS)
- CloudTrail: audit logging of all API calls
- GuardDuty: detective control, alerts on suspicious behavior
- One preventative control (IAM boundary or SCP)

## What was done in this session
- Read the full assignment PDF (Wiz_Tech_Exercise_V4.pdf at repo root)
- Created wiz-exercise/ folder structure (app/, terraform/, k8s/, .github/workflows/)
- Copied week07-movie-buddy code into wiz-exercise/app/ (week07 folder left untouched)
- Wrote Dockerfile (wiz-exercise/app/Dockerfile) — explained line by line
- Created wizexercise.txt with "Oleksii Androsov"
- Pushed everything to GitHub

## What to do next (start here)
1. git pull on MacBook Air to get the files
2. Verify Docker Desktop is installed and running (docker --version)
3. Rewrite wiz-exercise/app/memory.py — swap DynamoDB for MongoDB
4. Update wiz-exercise/app/requirements.txt
5. docker build and test locally
6. Then move to Terraform

## Key decisions made
- New wiz-exercise/ folder for all Wiz work — weekly folders untouched
- Using existing MovieBuddy app as the web application (own choice, allowed by assignment)
- AWS as cloud provider (Oleksii's strongest platform)
- movie-buddy.app domain will be re-pointed to new Load Balancer DNS

## File locations
- Assignment PDF: wiz-exercise/../Wiz_Tech_Exercise_V4.pdf (repo root)
- App code: wiz-exercise/app/
- Dockerfile: wiz-exercise/app/Dockerfile
- This file: wiz-exercise/HANDOFF.md

## Repo
https://github.com/oleksii-androsov/ai-learning
Branch: main
