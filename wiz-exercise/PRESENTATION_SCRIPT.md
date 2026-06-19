# Wiz Technical Exercise — Presentation Script
# 30 minutes presentation + 15 minutes Q&A

---

## Visual style guidance for all ChatGPT image prompts
- Dark navy background (#0a0e1a) for full-bleed slides
- Dark charcoal (#1e2433) for content cards and boxes
- Electric blue (#00B4D8) for arrows, accents, highlights
- Red (#FF4444) for warnings, security findings, critical items
- Green (#00C875) for resolved, positive, compliant items
- White text on dark backgrounds
- Flat vector illustration style — no 3D, no photo-realism, no gradients except subtle glows
- Clean, modern, enterprise security aesthetic inspired by Wiz.io visual language
- All images 16:9 widescreen format

---

## BEFORE YOU START — Meeting management checklist
- Ask for introductions round: "Before I dive in, I'd love to go around and hear who's joining and what you're most curious to see"
- Take notes as people answer
- Adjust depth of technical content based on audience
- Ask for feedback at the end
- Follow up by email after the call

---

## Slide 1 — Title (1 min)

**Speaker notes:**
"Good morning / afternoon everyone. Thank you for having me. Before I dive in — I'd love to quickly go around and hear who's joining today and what you're most curious to see. That way I can make sure I address what matters most to each of you."

*(Take notes as people answer)*

**Slide content:** Title: "Securing a Cloud-Native Application on AWS". Subtitle: "A DevSecOps Exercise — Oleksii Androsov"

**ChatGPT image prompt:**
Dark navy background (#0a0e1a). Centered composition. A stylized AWS cloud architecture diagram rendered as glowing neon lines on dark background — showing a Kubernetes cluster icon, a database cylinder, an S3 bucket, and a load balancer connected by flowing electric-blue (#00B4D8) lines. Flat vector illustration style, no gradients except subtle blue glow effects. No text. Modern enterprise security aesthetic inspired by Wiz.io. 16:9 widescreen format.

---

## Slide 2 — Agenda (1 min)

**Speaker notes:**
"Here's what we'll cover in the next 30 minutes. I'll start with the task I was given, then walk you through the solution I built and a live demo of the running application. From there we'll look at the DevSecOps pipeline, the security findings from the AWS native tools, and how Wiz would provide value in this environment. I'll close with what I'd do differently and what I took away from this exercise. We'll have 15 minutes for questions at the end — but feel free to interrupt at any point if something sparks a question."

**Slide content:** Numbered agenda list:
1. The Task
2. The Solution — Architecture & Live Demo
3. How I Built It — DevSecOps Pipeline
4. Security Findings
5. The Value of Wiz
6. What I'd Do Differently & What I Learned
7. Q&A

**ChatGPT image prompt:**
Minimal vertical agenda layout on dark navy background (#0a0e1a). Seven numbered items in a clean vertical list, each on its own row. Numbers in large electric blue (#00B4D8) circles on the left, item text in white on the right. Items: "1 The Task", "2 The Solution & Live Demo", "3 DevSecOps Pipeline", "4 Security Findings", "5 The Value of Wiz", "6 What I'd Do Differently & Learned", "7 Q&A". Row 7 slightly muted/lighter to indicate it's after the main presentation. Clean sans-serif typography, generous spacing between rows. No decorative elements. 16:9 widescreen.

---

## Slide 3 — The Task (2 min)

**Speaker notes:**
"Let me start with what I was actually asked to do — because the task itself is interesting and explains every architectural decision you'll see.

The assignment was to deploy a two-tier cloud application — a containerized web app backed by a MongoDB database — using modern DevOps practices. But with a twist: several components had to be deliberately misconfigured. The database server had to run an outdated version of MongoDB on an outdated Linux OS, with SSH exposed to the internet and an overly permissive cloud role. The storage bucket for database backups had to be publicly readable.

The containerized application had to be assigned a Kubernetes cluster-admin role — which in practice means the running container has full control over the entire Kubernetes cluster.

On top of that, I had to build a full CI/CD pipeline with security scanning at each stage, and implement AWS-native security controls to detect the misconfigurations.

The goal of all of this is to simulate the kind of environment that security teams encounter in the real world — where applications are built under delivery pressure, security is treated as an afterthought, and the attack surface is much larger than anyone realizes. And then to demonstrate how you'd find and communicate those risks."

**Slide content:** Task summary with key requirements highlighted

**ChatGPT image prompt:**
Split two-column layout on dark navy background (#0a0e1a). Left column header: "The Requirements" — bulleted list items in white with small electric blue bullet dots: "Containerized app on Kubernetes", "MongoDB on EC2 (outdated)", "CI/CD pipeline with security scanning", "AWS-native security controls". Right column header: "Intentional Weaknesses" — same list style but bullet dots are red warning triangles: "SSH open to internet", "Overly permissive IAM role", "Public S3 bucket", "Cluster-admin pod role", "Outdated OS + DB versions". Both columns in dark charcoal (#1e2433) rounded card. Thin electric blue border on left column, thin red border on right column. White text throughout. Clean, minimal. 16:9 widescreen.

---

## Slide 4 — The Solution — Architecture (2 min)

**Speaker notes:**
"Here's what I built. The application is called MovieBuddy — an AI-powered movie recommendation chatbot. I chose it because it's a genuinely useful multi-agent system rather than a toy app, which makes the demo more meaningful. It uses Claude Sonnet as the AI backbone, Tavily for live web search, and TMDB for movie posters.

Architecturally: traffic comes in through an AWS Application Load Balancer with HTTPS termination via ACM certificates. The load balancer routes to a Kubernetes pod running on EKS — in private subnets. The pod connects to MongoDB on an EC2 instance for persistent user memory. Database backups run daily to S3.

The red warning icons you can see on the diagram are the intentional weaknesses — we'll come back to each of them when we look at the security findings."

*(Point to each component on the diagram as you describe it)*

**Slide content:** Architecture diagram image

**ChatGPT image prompt:**
Clean AWS architecture diagram on dark navy background (#0a0e1a). Show left to right: Internet user icon → DNS/Route53 → Application Load Balancer (with padlock/SSL icon) → EKS cluster box (containing a Kubernetes pod icon labeled "MovieBuddy") → MongoDB cylinder on EC2 instance → S3 bucket for backups. Use AWS-style flat service icons. Electric blue (#00B4D8) connection arrows, white labels. Add small red warning triangle icons on: EC2 (SSH open to internet), S3 (public bucket), IAM badge on EC2 (overpermissive), Kubernetes pod (cluster-admin). Flat 2D diagram, clean lines, no shadows. 16:9 widescreen.

---

## Slide 5 — Live Demo — The Application (5 min)

**Speaker notes:**
"Let me show you the application live."

*(Switch to browser — open https://movie-buddy.app)*

"This is MovieBuddy — running live on Kubernetes, accessible over HTTPS. It's a multi-agent AI system: an Orchestrator coordinates four specialist agents — a Tracker, an Explorer, a Fact-Checker, and a Planner — running in parallel. It uses Claude Sonnet as the AI backbone, Tavily for live web search, and TMDB for movie posters."

*(Have a conversation — e.g. "What's a good family film for a rainy evening with kids aged 8 and 12?" — show the response)*

"User profiles are stored persistently in MongoDB — so when you come back, it remembers your preferences."

*(Show the sidebar profile panel)*

"Let me prove the data is actually in the database."

*(Run in terminal)*
```bash
kubectl exec -it $(kubectl get pod -l app=movie-buddy -o jsonpath='{.items[0].metadata.name}') \
  -- python3 -c "
from pymongo import MongoClient
import os, json
c = MongoClient(os.environ['MONGO_URL'])
docs = list(c.movie_buddy.profiles.find({}, {'_id': 0}))
print(json.dumps(docs, indent=2, default=str))
"
```

"You can see the profile data — genres, movies watched, kids ages — all persisted in MongoDB."

**Slide content:** App mockup image + 3 bullet points: "Multi-agent AI (Claude Sonnet)" | "EKS on AWS, HTTPS" | "MongoDB persistent memory"

**ChatGPT image prompt:**
Dark-themed chat UI mockup illustration. Split composition: left two-thirds shows a chat interface with dark background (#1e2433), a conversation about movie recommendations, movie poster thumbnails arranged in a row, and four colored specialist agent badges at the top (blue "Tracker", green "Explorer", orange "Fact-Checker", purple "Planner"). Right one-third shows a sidebar panel with user profile data — genre preferences, kids ages, streaming platforms listed. Electric blue and purple accent colors. Clean sans-serif typography feel. Flat design illustration, not a screenshot. 16:9 widescreen.

---

## Slide 6 — Business Benefits and Risks (3 min)

**Speaker notes:**
"Before we look at security in detail, let me frame the business context — because security findings only matter if you understand what they mean for the business.

This environment has clear value: it's fully automated, reproducible, scalable. I can tear it down and redeploy the entire thing in about 30 minutes from a single Terraform command. That's real business agility.

But the way it was built introduces risks with direct business impact:

SSH open to the internet on the database server — this means any attacker, anywhere, can attempt to brute-force access to the machine that holds your customer data. There's no IP restriction, no VPN requirement.

The EC2 instance has AdministratorAccess IAM role attached. If an attacker compromises that machine — through the open SSH, through a MongoDB vulnerability, any way — they immediately inherit full control of the entire AWS account. They can create new users, exfiltrate data, spin up crypto mining infrastructure, delete everything. One compromised VM becomes a compromised cloud account.

The S3 bucket that holds database backups is publicly readable. And as we'll see shortly, it also contains something it shouldn't.

Both MongoDB and the operating system are over a year out of date — meaning there are known, documented attack techniques for these versions available publicly."

**Slide content:** Two columns — Benefits | Risks

**ChatGPT image prompt:**
Two-panel infographic on dark navy background (#0a0e1a). Left panel labeled "Business Value" — three items each with a green checkmark circle icon: "Automated Deployment", "Horizontally Scalable", "Full Audit Trail". Right panel labeled "Security Risks" — four items each with a red warning triangle icon: "SSH Exposed to Internet", "Over-Privileged IAM Role", "Credentials in Public S3", "Unpatched CVEs". Vertical electric blue (#00B4D8) dividing line between panels. White text, flat icon style. No decorative elements. Clean enterprise look. 16:9 widescreen.

---

## Slide 7 — How I Built It — DevSecOps Pipeline — LIVE DEMO (5 min)

**Speaker notes:**
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

**Slide content:** Pipeline flow diagram image

**ChatGPT image prompt:**
Horizontal pipeline flow diagram on dark navy background (#0a0e1a). Two parallel pipeline tracks stacked vertically, each with a label on the left. Top track labeled "Infra Pipeline": connected boxes left to right — "Git Push" → "Checkov Scan" (red security shield icon) → "Terraform Plan" (document icon) → "Manual Approval" (human/pause icon, amber color) → "Terraform Apply" → "AWS" (cloud icon). Bottom track labeled "App Pipeline": "Git Push" → "Trivy Scan" (red shield) → "Docker Build" → "ECR Push" → "kubectl rollout" → "EKS" (Kubernetes wheel icon). Electric blue (#00B4D8) arrows connecting boxes. Boxes in dark charcoal (#1e2433) with white text. Security scan steps have red accent border, deploy steps have green accent border. Flat vector style. 16:9 widescreen.

---

## Slide 8 — Security Findings — LIVE DEMO (8 min)

**Speaker notes:**
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

**Slide content:** Security findings dashboard image

**ChatGPT image prompt:**
Security findings dashboard on dark navy background (#0a0e1a). Five tool labels as column headers in a row across the top: "Checkov", "Trivy", "AWS Inspector", "GuardDuty", "AWS Config / CloudTrail". Below each header, 2 finding cards in dark charcoal (#1e2433). Cards have colored left-border severity indicator: red = CRITICAL, orange = HIGH. Finding labels in white text: under Checkov — "SSH open 0.0.0.0/0", "AdministratorAccess IAM"; under Trivy — "Base image CVEs"; under Inspector — "MongoDB 4.4 CVEs", "Ubuntu 20.04 EOL"; under GuardDuty — "Runtime monitoring active"; under Config/CloudTrail — "Config drift detection", "Full API audit log". Bottom center: large badge "10 Findings Detected" in red. Flat design, no gradients. 16:9 widescreen.

---

## Slide 9 — What Value Would Wiz Provide — LIVE DEMO (5 min)

**Speaker notes:**
"That question is exactly the right one — and none of the tools I just showed you can answer it. Let me show you why.

To get a complete picture of this environment, I checked six different tools: Checkov, Trivy, Inspector, GuardDuty, Config, CloudTrail. Six interfaces, six finding formats, six places to look. And after all of that — I still don't have a prioritized answer to: what do I fix first, and why?"

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

"The Terraform state file contains the MongoDB connection string — username and password — in plain text. And it's sitting in a publicly accessible bucket. Anyone who knows this bucket name can read this file and connect directly to the database. No credentials required at any step.

That's a four-step attack path: public S3 bucket → readable state file → exposed credentials → full database access. Checkov flagged the public bucket as one finding. It flagged missing encryption as another. But no tool connected these dots and said: together, these create a complete path from the public internet to your customer data.

This is precisely what Wiz was built to solve. Wiz builds a graph of your entire cloud environment — every resource, every permission, every network path — and uses that graph to identify which combinations of findings create real attack paths. Instead of ten isolated findings, you get three prioritized attack paths ranked by actual exploitability and business impact. Your security team stops triaging alerts and starts fixing what actually matters.

In CI/CD pipelines, you can add tools like Semgrep for code scanning, KICS for additional IaC checks. Each adds signal — but also another pane of glass. Wiz sits above all of these and gives you the unified view with context to act."

**Slide content:** Fragmentation vs unified view image

**ChatGPT image prompt:**
Split composition on dark navy background (#0a0e1a). Left half labeled "Without Wiz" (muted, slightly gray tint): Six isolated tool icons scattered — Checkov, Trivy, Inspector, GuardDuty, Config, CloudTrail — each in its own separate dark box with no connections between them, red warning icons floating disconnected, chaotic arrangement suggesting fragmentation and alert fatigue. Right half labeled "With Wiz" (vivid, full color): A clean connected graph — five nodes labeled "Public Internet" → "Public S3 Bucket" → "Terraform State File" → "MongoDB Credentials" → "Database" — connected by a glowing red attack path arrow showing the full chain. Below the graph: "1 Critical Attack Path" in bold red. Vertical dividing line in electric blue (#00B4D8) between the two halves. Flat vector style. 16:9 widescreen.

---

## Slide 10 — Challenges & What I Learned (3 min)

**Speaker notes:**
"Building this wasn't without friction — and the friction is where the real learning happened. Let me walk through both.

Four technical challenges I hit:

First — cross-platform Docker builds. My Mac runs ARM64, EKS runs AMD64. The container built fine locally and crashed immediately on Kubernetes. Fixed with docker buildx and the --platform linux/amd64 flag. This taught me that container portability isn't automatic — you have to be explicit about target architecture.

Second — the AWS Load Balancer Controller couldn't discover my subnets because they were missing required Kubernetes cluster tags. I had to add those tags via Terraform and explicitly pin subnet IDs in the ingress. This taught me that the ALB controller has very specific expectations about subnet tagging, and when those aren't met, the error messages are not obvious.

Third — Streamlit uses WebSockets for its live UI. Behind an ALB without sticky sessions, every page refresh broke the connection. Fixed with sticky session annotations on the ingress. This taught me that not all HTTP traffic behaves the same — WebSocket applications have specific load balancer requirements.

Fourth — Terraform state. The CI/CD pipeline had no access to state on my laptop. Migrating to S3 made the pipeline work — but also revealed a security issue: the state file contains credentials in plain text, and I'd put it in a public bucket. That mistake became the strongest demo moment in the presentation.

What I took away from the exercise overall: the most interesting security findings in this environment weren't the obvious ones — open SSH, outdated software. The most dangerous finding emerged from the combination of two seemingly unrelated decisions: a public S3 bucket and Terraform remote state. Neither decision is alarming on its own. Together they create a complete attack path. That's the insight that made the Wiz value proposition click for me — it's not about finding more things, it's about connecting the things you already know."

**Slide content:** Four challenge cards + key learning callout

**ChatGPT image prompt:**
Two-section layout on dark navy background (#0a0e1a). Top section: four-card horizontal row — cards in dark charcoal (#1e2433) rounded rectangles. Card 1: Docker whale icon — "ARM→AMD64 Builds". Card 2: network icon — "ALB Subnet Tags". Card 3: lightning icon — "WebSocket + Sticky Sessions". Card 4: database/cloud icon — "Terraform Remote State". Each card has a small amber dot top-right (challenge) and green checkmark bottom-right (resolved). Electric blue border on each card. Bottom section: single wide callout card with electric blue left border and slightly lighter background. Bold white text: "Key Insight" label, then smaller text: "The most dangerous finding came from combining two unrelated decisions — not from any single misconfiguration." Flat vector style, white text. 16:9 widescreen.

---

## Slide 11 — What Would You Do Differently (2 min)

**Speaker notes:**
"If I were hardening this for production, four changes in priority order:

First — AWS Secrets Manager with the External Secrets Operator. Remove credentials from Kubernetes Secrets and Terraform state entirely. Secrets live in AWS, rotated automatically, never stored in plain text anywhere.

Second — IMDSv2 enforcement on EC2 and immutable ECR image tags. IMDSv2 prevents server-side request forgery attacks from stealing instance credentials. Immutable tags prevent image tampering after a scan passes.

Third — replace static AWS credentials in GitHub Secrets with OIDC federation. The pipeline assumes an IAM role dynamically — no long-lived credentials stored anywhere.

Fourth — move MongoDB to a private subnet, remove the public IP from EC2, add VPC endpoints. Database traffic never leaves the AWS network."

**Slide content:** Four-step roadmap image

**ChatGPT image prompt:**
Four-step improvement roadmap on dark navy background (#0a0e1a). Horizontal timeline with four numbered circular nodes (1, 2, 3, 4) connected by a progress bar line. Color gradient from amber/orange on the left (current state) to bright green on the right (target state). Node 1 (amber): key/lock icon — "Secrets Manager + ESO" label below. Node 2: shield icon — "IMDSv2 + Immutable ECR". Node 3: chain/link icon — "OIDC Federation". Node 4 (green): network diagram icon — "Private Subnet + VPC Endpoints". Below each node, a small dark charcoal (#1e2433) card with 2-line description. Above the timeline: "Current State" label on far left, "Production Ready" on far right. Flat vector style, white text. 16:9 widescreen.

---

## Slide 12 — Close / Q&A (1 min)

**Speaker notes:**
"That brings me to the end of the presentation. To summarise: I built a real, working cloud-native application, deliberately introduced the security weaknesses the exercise required, built a DevSecOps pipeline that surfaces those weaknesses automatically, and implemented AWS-native controls for detection and audit. And I showed you how the fragmentation of findings across those tools is exactly the problem Wiz was built to solve.

I'd love to open it up for questions — and I'd genuinely appreciate your feedback. Is there anything you'd like me to go deeper on, or anything you'd have done differently?"

*(After Q&A, close with:)*
"Thank you all for your time. I'll follow up by email with the GitHub repository link. Looking forward to the next steps."

**Slide content:** "Thank you" + GitHub repo URL + contact email

**ChatGPT image prompt:**
Minimal closing slide on dark navy background (#0a0e1a). Center-aligned composition. Large white text "Thank You" at top. Below, two lines in electric blue (#00B4D8): a GitHub icon with "github.com/oleksii-androsov/ai-learning" and an envelope icon with "oleksiiandrosov85@gmail.com". Below those, in smaller white text: "Questions & Discussion". Subtle glowing electric blue horizontal line separating "Thank You" from the contact details. Clean, understated, professional. No other decorative elements. 16:9 widescreen.

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

# Show MongoDB backup files in S3
aws s3 ls s3://movie-buddy-db-backups/

# Show Terraform state with credentials (attack path demo)
aws s3 cp s3://movie-buddy-tfstate-329153220664/wiz-exercise/terraform.tfstate - | python3 -m json.tool | grep -A3 mongodb_url

# Show S3 bucket is publicly accessible (open in browser, no login)
# https://movie-buddy-db-backups.s3.amazonaws.com/

# SSH to MongoDB server (demonstrate open SSH)
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237

# Check MongoDB and OS version
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237 "mongod --version && lsb_release -a"
```
