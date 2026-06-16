# Wiz Technical Exercise — Presentation Script
# 30 minutes presentation + 15 minutes Q&A

---

## BEFORE YOU START — Meeting management checklist
- Ask for introductions round: "Before I dive in, I'd love to go around and hear who's joining and what you're most curious to see"
- Take notes as people answer
- Adjust depth of technical content based on audience
- Ask for feedback at the end
- Follow up by email after the call

---

## Slide 1 — Title (1 min)

"Good morning / afternoon everyone. Thank you for having me. Before I dive in — I'd love to quickly go around and hear who's joining today and what you're most curious to see. That way I can make sure I address what matters most to each of you."

*(Take notes as people answer)*

---

## Slide 3 — Overview / Architecture (3 min)

"The exercise asked me to deploy a real, working cloud application — deliberately configured with security weaknesses that you'd realistically find in enterprise environments built under delivery pressure.

I chose to build MovieBuddy: an AI-powered movie recommendation chatbot that uses Claude and real-time web search. It's a multi-agent system — not a toy app. Here's the architecture."

*(Point to architecture diagram)*

"The application runs in a containerized Kubernetes cluster on EKS — in private subnets. Traffic comes in through an AWS Application Load Balancer with HTTPS termination via ACM certificates. The database is MongoDB running on EC2 — and I've built in several intentional security weaknesses that we'll get to shortly. The whole infrastructure is defined as code in Terraform and deployed via GitHub Actions CI/CD pipelines."

---

## Slide 4 — What You Built — LIVE DEMO (5 min)

"Let me show you the application live."

*(Switch to browser — open https://movie-buddy.app)*

"This is MovieBuddy — running live on Kubernetes, accessible over HTTPS. It's a multi-agent AI system: an Orchestrator coordinates four specialist agents — a Tracker, an Explorer, a Fact-Checker, and a Planner — running in parallel. It uses Claude Sonnet as the AI backbone, Tavily for live web search, and TMDB for movie posters."

*(Have a conversation — e.g. "What's a good family film for a rainy evening with kids aged 8 and 12?" — show the response)*

"User profiles are stored persistently in MongoDB — so when you come back, it remembers your preferences."

*(Show the sidebar profile panel)*

"Let me prove the data is actually in the database."

*(Run in terminal)*
```
kubectl exec -it $(kubectl get pod -l app=movie-buddy -o jsonpath='{.items[0].metadata.name}') \
  -- python3 -c "
from pymongo import MongoClient
import os
c = MongoClient(os.environ['MONGO_URL'])
import json
docs = list(c.movie_buddy.profiles.find({}, {'_id': 0}))
print(json.dumps(docs, indent=2, default=str))
"
```

"You can see the profile data — genres, movies watched, kids ages — all persisted in MongoDB."

---

## Slide 5 — Business Benefits and Risks (3 min)

"Before we look at security in detail, let me frame the business context — because security findings only matter if you understand what they mean for the business.

This environment has clear value: it's fully automated, reproducible, scalable. I can tear it down and redeploy the entire thing in about 30 minutes from a single Terraform command. That's real business agility.

But the way it was built introduces risks with direct business impact:

SSH open to the internet on the database server — this means any attacker, anywhere, can attempt to brute-force access to the machine that holds your customer data. There's no IP restriction, no VPN requirement.

The EC2 instance has AdministratorAccess IAM role attached. If an attacker compromises that machine — through the open SSH, through a MongoDB vulnerability, any way — they immediately inherit full control of the entire AWS account. They can create new users, exfiltrate data, spin up crypto mining infrastructure, delete everything. One compromised VM becomes a compromised cloud account.

The S3 bucket that holds database backups is publicly readable. And as we'll see shortly, it also contains something it shouldn't.

And both MongoDB and the operating system are over a year out of date — meaning there are known, documented attack techniques for these versions available publicly."

---

## Slide 6 — How You Built It — LIVE DEMO (5 min)

"I built this using a full DevSecOps pipeline. Let me show you how it works."

*(Switch to GitHub — show repo structure briefly)*

"Everything lives in GitHub. Infrastructure code in the infra folder, application code in the app folder, Kubernetes manifests in k8s, and two GitHub Actions workflows."

*(Switch to GitHub Actions — show both workflows)*

"The infrastructure pipeline triggers whenever I push a change to a Terraform file. It runs Checkov — an IaC security scanner — then generates a Terraform plan showing exactly what will change in AWS, pauses for manual approval, and only then applies the changes. No infrastructure change can reach production without a human reviewing both the security findings and the plan."

*(Click into the Infra CI run — show Checkov findings, show the approval gate)*

"The application pipeline triggers on code changes. Trivy scans the container for vulnerabilities, Docker builds the image for the correct platform — AMD64 for EKS — pushes it to ECR, and kubectl rolls out the new version to EKS automatically."

*(Click into App CI/CD run)*

"Secrets are in two places: CI/CD credentials live in GitHub Secrets — never in the code. Application secrets — API keys and the MongoDB connection string — live in a Kubernetes Secret and are injected into the pod as environment variables at startup. Let me show you."

```bash
kubectl get secrets
kubectl get secret movie-buddy-secrets -o jsonpath='{.data}' | python3 -m json.tool
```

---

## Slide 7 — Challenges Faced (2 min)

"Building this wasn't without friction. Four real challenges worth mentioning:

First — cross-platform Docker builds. My Mac runs on ARM64, EKS runs AMD64. The container would build fine locally and crash immediately on Kubernetes. Fixed with docker buildx and the --platform linux/amd64 flag.

Second — the AWS Load Balancer Controller couldn't discover my subnets automatically because they were missing required Kubernetes cluster tags. I had to add those tags via Terraform and explicitly pin the subnet IDs in the ingress configuration.

Third — Streamlit uses WebSockets for its live UI updates. Behind an ALB without sticky sessions, every page refresh would land on a different connection and the app would fail to load. Fixed by adding sticky session annotations to the Kubernetes ingress.

Fourth — Terraform state. The CI/CD pipeline has no access to a state file on my laptop. I migrated state to S3, which makes it accessible to any pipeline runner and enables true infrastructure automation."

---

## Slide 8 — Security Findings — LIVE DEMO (8 min)

"Now the most important part — what did the security tools actually find. I'll walk through each one live."

*(Switch to GitHub Actions → Infra CI run → Checkov step)*

"First, Checkov — this runs automatically every time I push infrastructure code, before anything touches AWS. It found 10 findings. The two most critical: SSH open to the entire internet on the EC2 instance, and AdministratorAccess IAM role attached to that same machine. In a real environment, these would block the pipeline — here I've set it to soft-fail intentionally so we can discuss the findings."

*(Switch to AWS Console → Inspector)*

"AWS Inspector scanned both the EC2 instance and the container images in ECR. It found critical CVEs in MongoDB 4.4 and Ubuntu 20.04 — both end-of-life, both with known unpatched vulnerabilities. What does that mean in practice? There are documented attack techniques for these versions publicly available — any attacker can look them up. Inspector gives me severity scores and CVE links, but it doesn't tell me whether these vulnerabilities are actually reachable from outside."

*(Switch to AWS Console → GuardDuty)*

"GuardDuty is runtime threat detection. It monitors CloudTrail API logs, VPC flow logs, and DNS queries, looking for patterns that indicate malicious behavior — unusual API calls, connections to known malicious IPs, credential abuse. It's detective rather than preventative — it doesn't stop an attack, but tells you one is happening. In a mature environment you'd pipe these findings into a SIEM and trigger automated response."

*(Switch to AWS Console → CloudTrail)*

"CloudTrail gives me a complete audit log of every API call in this account — who did what, when, from which IP. Every Terraform apply, every kubectl command, every ECR push. This is your forensic trail. If something goes wrong, this is where the investigation starts."

*(Switch to AWS Console → Config)*

"AWS Config tracks configuration state over time. If someone manually changes a security group — bypassing the pipeline — Config records the change, who made it, and what it looked like before. This is how you detect configuration drift between what your IaC defines and what's actually running."

"So across five tools, I have findings. But here's the question — which one do I fix first? And do any of these findings connect to each other in a way that creates a bigger risk than each one individually?"

---

## Slide 9 — What Value Would Wiz Provide — LIVE DEMO (5 min)

"That question is exactly the right one — and none of the tools I just showed you can answer it. Let me show you why."

"To get a complete picture of this environment, I checked six different tools: Checkov, Trivy, Inspector, GuardDuty, Config, CloudTrail. Six interfaces, six finding formats, six places to look. And after all of that — I still don't have a prioritized answer to: what do I fix first, and why?"

"Let me show you something concrete."

*(Open browser — navigate to the S3 bucket directly, no login)*
```
https://movie-buddy-db-backups.s3.amazonaws.com/
```

"This bucket is publicly readable — no credentials required. That's a Checkov finding. Now watch what's inside."

*(Run in terminal)*
```bash
aws s3 cp s3://movie-buddy-tfstate-329153220664/wiz-exercise/terraform.tfstate - \
  | python3 -m json.tool | grep -A3 mongodb_url
```

"The Terraform state file contains the MongoDB connection string — username and password — in plain text. And it's sitting in a publicly accessible bucket. Anyone who knows this bucket name can read this file and connect directly to the database. No credentials required at any step."

"That's a four-step attack path: public S3 bucket → readable state file → exposed credentials → full database access. Checkov flagged the public bucket as one finding. It flagged missing encryption as another. But no tool connected these dots and said: together, these create a complete path from the public internet to your customer data."

"This is precisely what Wiz was built to solve. Wiz builds a graph of your entire cloud environment — every resource, every permission, every network path — and uses that graph to identify which combinations of findings create real attack paths. Instead of ten isolated findings, you get three prioritized attack paths ranked by actual exploitability and business impact. Your security team stops triaging alerts and starts fixing what actually matters."

"In CI/CD pipelines, you can add tools like Semgrep for code scanning, KICS for additional IaC checks. Each adds signal — but also another pane of glass. Wiz sits above all of these and gives you the unified view with context to act."

---

## Slide 10 — What Would You Do Differently (2 min)

"If I were hardening this for production, four changes in priority order:

First — AWS Secrets Manager with the External Secrets Operator. Remove credentials from Kubernetes Secrets and Terraform state entirely. Secrets live in AWS, rotated automatically, never stored in plain text anywhere.

Second — IMDSv2 enforcement on EC2 and immutable ECR image tags. IMDSv2 prevents server-side request forgery attacks from stealing instance credentials. Immutable tags prevent image tampering after a scan passes.

Third — replace static AWS credentials in GitHub Secrets with OIDC federation. The pipeline assumes an IAM role dynamically — no long-lived credentials stored anywhere.

Fourth — move MongoDB to a private subnet, remove the public IP from EC2, add VPC endpoints. Database traffic never leaves the AWS network."

---

## Slide 11 — Resources + Close (1 min)

"Happy to share all of these links by email after the call."

*(Close with)*

"Thank you all — this was a genuinely enjoyable exercise to build. I'd love to hear your feedback: is there anything you'd have liked to see more of, or any area where you'd want me to go deeper? I'll send a follow-up with the GitHub repo link and resources."

---

## DEMO CHEAT SHEET — Commands to have ready in terminal

```bash
# Show running pod
kubectl get pods -o wide

# Show all K8s objects
kubectl get deployment,service,ingress,clusterrolebinding | grep movie-buddy

# Show ingress details (ALB, cert, annotations)
kubectl describe ingress movie-buddy-ingress

# Show secrets (names only, values hidden)
kubectl get secret movie-buddy-secrets -o jsonpath='{.data}' | python3 -m json.tool

# Show wizexercise.txt
kubectl exec -it $(kubectl get pod -l app=movie-buddy -o jsonpath='{.items[0].metadata.name}') -- cat /app/wizexercise.txt

# Show data in MongoDB
kubectl exec -it $(kubectl get pod -l app=movie-buddy -o jsonpath='{.items[0].metadata.name}') -- python3 -c "
from pymongo import MongoClient; import os, json
c = MongoClient(os.environ['MONGO_URL'])
docs = list(c.movie_buddy.profiles.find({}, {'_id': 0}))
print(json.dumps(docs, indent=2, default=str))
"

# Show MongoDB backup files
aws s3 ls s3://movie-buddy-db-backups/

# Show Terraform state with credentials (attack path demo)
aws s3 cp s3://movie-buddy-tfstate-329153220664/wiz-exercise/terraform.tfstate - | python3 -m json.tool | grep -A3 mongodb_url

# SSH to MongoDB server (show open SSH)
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237

# Check MongoDB version
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237 "mongod --version && lsb_release -a"
```
