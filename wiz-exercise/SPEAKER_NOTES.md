# Speaker Notes — Clean Copy for PowerPoint
# Paste each section into the Notes panel of the corresponding slide

---

## Slide 1 — Title

Good morning / afternoon everyone. Thank you for having me. Before I dive in — I'd love to quickly go around and hear who's joining today and what you're most curious to see. That way I can make sure I address what matters most to each of you.

[Take notes as people answer]

---

## Slide 2 — Agenda

Here's the plan for the next 30 minutes. The task I was given, then the architecture and a live demo. From there the DevSecOps pipeline, the security findings, and how Wiz would change the picture. I'll close with what I'd do differently and what I learned. Questions at the end — but feel free to interrupt.

---

## Slide 3 — The Task

Let me start with what I was actually asked to do — because the task explains every architectural decision you'll see.

Deploy a two-tier cloud application — a containerised web app backed by MongoDB — using modern DevOps practices. But with deliberate misconfigurations: outdated MongoDB on an outdated OS, SSH exposed to the internet, an overly permissive cloud role, a publicly readable S3 bucket, and a Kubernetes pod with cluster-admin rights — meaning the running container has full control of the entire cluster.

On top of that: a full CI/CD pipeline with security scanning at each stage, and AWS-native security controls to detect the misconfigurations.

The goal is to simulate what security teams encounter in the real world — applications built under delivery pressure, security as an afterthought, attack surface larger than anyone realises. And then show how you'd find and communicate those risks.

---

## Slide 4 — Architecture

Here's what I built. The application is called MovieBuddy — an AI-powered movie recommendation chatbot. I chose a real multi-agent system rather than a toy app so the demo is more meaningful.

Traffic comes in through an AWS Application Load Balancer with HTTPS via ACM certificates. The load balancer routes to a Kubernetes pod on EKS in private subnets. The pod connects to MongoDB on EC2 for persistent user memory. Daily backups go to S3.

The red warning icons on the diagram are the intentional weaknesses — we'll come back to each of them.

---

## Slide 5 — Live Demo

[Switch to browser — open https://movie-buddy.app]

This is MovieBuddy — running on Kubernetes, over HTTPS. Four specialist AI agents working in parallel, Claude Sonnet as the backbone, live web search via Tavily, movie posters from TMDB.

[Ask: "What's a good family film for a rainy evening with kids aged 8 and 12?"]

Preferences are stored in MongoDB so the app remembers you. Let me prove the data is actually there.

[Run kubectl exec command — show profile data]

Profile data, persisted in MongoDB. Now let me show you what's running under the hood.

[Run: kubectl get pods -o wide]

[Switch to AWS Console — EC2 → Instances → movie-buddy-mongodb]

The MongoDB server — EC2 in a public subnet, public IP, Ubuntu 20.04. Note the public subnet and public IP. We'll come back to why that matters.

---

## Slide 6 — DevSecOps Pipeline

I built this using a full DevSecOps pipeline. Let me show you how it works.

[Switch to GitHub — show repo structure briefly]

Everything lives in GitHub. Infrastructure code in infra, application code in app, Kubernetes manifests in k8s, two GitHub Actions workflows.

[Switch to GitHub Actions — show both workflows]

The infrastructure pipeline triggers on Terraform file changes. Checkov runs first — IaC security scanner — then Terraform plan showing exactly what will change, then manual approval, then apply. No infrastructure change reaches production without a human reviewing both security findings and the plan.

[Click into Infra CI run — show Checkov findings, show approval gate]

The app pipeline builds the Docker image first, then Trivy scans it before it goes anywhere near ECR. 14 vulnerabilities found — two critical CVEs in perl, high-severity in ncurses and SQLite. Because we output SARIF format, GitHub renders them in the Security tab.

[Switch to GitHub repo → Security → Code scanning — show briefly]

Findings directly where developers work. In production you'd set exit-code 1 — push blocked until resolved.

One more thing to flag: Terraform remote state. Terraform keeps a record of everything it's deployed — resource IDs, addresses, and any credentials it used. When the pipeline runs in GitHub, it can't access my laptop, so I moved that file to S3. Right call for CI/CD. But I made the bucket public — and that state file contains database credentials in plain text. We'll come back to that.

[IF ASKED — why does tfstate contain credentials?]
Terraform stores credentials so it can detect drift — compare what's in code against what's actually deployed. It doesn't encrypt state by default. Known limitation.

[IF ASKED — why S3 and not GitHub-native?]
GitHub has no built-in Terraform state storage. S3 is the standard AWS choice. The better option is Terraform Cloud, which encrypts state and controls access. The mistake wasn't S3 — it was making the bucket public.

Secrets live in two places: AWS credentials for the pipeline are in GitHub Secrets, never in code. Application secrets — API keys and the MongoDB URL — are in a Kubernetes Secret, injected as environment variables at pod startup.

---

## Slide 7 — Business Benefits and Risks

Before we look at findings — the business context.

This environment deploys in 30 minutes from a single Terraform command. Fully automated, reproducible, horizontally scalable. That's real value.

But it was built with five intentional weaknesses — and I want to show they're real, not hypothetical.

[Run: aws ec2 describe-security-groups — show SSH open to 0.0.0.0/0]
[Run: aws iam list-attached-role-policies — show AdministratorAccess]
[Run: kubectl get clusterrolebinding movie-buddy-cluster-admin -o yaml]

Open SSH — anyone on the internet can try to get in. AdministratorAccess — one compromised VM equals a compromised AWS account. Cluster-admin on the pod — code execution in the container means full control of Kubernetes. Plus a public S3 bucket and an OS and database both over a year out of date. Let's see what the security tools made of this.

---

## Slide 8 — Security Findings

Five tools, five angles. Let me walk through what each one found.

[Switch to GitHub Actions → Infra CI → Checkov step]

Checkov runs before anything touches AWS. 10 findings — the two that matter most: SSH open to the internet, AdministratorAccess on EC2. Soft-fail here intentionally so the pipeline still ran — in production these would block it.

[Switch to AWS Console → Inspector → filter by EC2, Critical]

Inspector scanned the EC2 and found 26 critical CVEs. Not 'Ubuntu is old' as a single flag — it lists every package with a known vulnerability. The one I want to highlight:

[Click into libssh / libssh-4 finding]

Critical CVE in libssh — the SSH library — on the same machine where port 22 is open to the internet. Not just the port is exposed, but the SSH implementation itself is vulnerable.

[Switch to Inspector → filter by ECR Container Image]

ECR: critical CVE in perl in the running container image. Inspector gives CVE IDs and severity scores but can't tell you if these are actually reachable from outside.

[Switch to AWS Console → GuardDuty → Findings]

GuardDuty watches for suspicious behavior — not misconfigurations, but actions. Two findings: public access was granted to the S3 bucket, and Block Public Access was disabled. Same bucket that Checkov and Config flagged — but GuardDuty caught the act of making it public. Three tools, same bucket, three completely different signals. None talking to each other.

[Switch to AWS Console → Config → Rules]

Config runs two compliance rules I deployed via Terraform. SSH open to internet: non-compliant. Public S3 bucket: non-compliant. Same findings as Checkov — but Config catches them on the live resource continuously, even if someone bypasses the pipeline and changes things manually.

[Switch to AWS Console → CloudTrail]

CloudTrail is the audit log — every API call, who made it, when, from which IP. If something goes wrong, this is where the investigation starts.

Five tools, 40+ findings. The question is — which one do I fix first? And do any of these connect into something worse than each one individually?

---

## Slide 9 — The Value of Wiz

None of those tools can answer that question. Let me show you why — concretely.

[Open browser — navigate to https://movie-buddy-tfstate-472151629584.s3.amazonaws.com/]

This is the Terraform state bucket. Publicly readable — no credentials, no AWS account. Just a browser.

[Run: aws s3 cp ... terraform.tfstate | grep -A3 mongodb_url]

Username and password in plain text. Now watch how far this goes.

[Run: ssh -i wiz-exercise/wiz-exercise-key ubuntu@44.201.2.128]

Port 22 is open to the internet. I'm in. I dump the entire database — and I don't need my own AWS credentials. The EC2 has AdministratorAccess. I use the machine's own identity.

[Run mongodump inside EC2, then aws s3 cp to exfiltrate]

[Back in local terminal: aws s3 ls s3://movie-buddy-tfstate-472151629584/exfil/]

The entire customer database is in S3 — staged using your own cloud account. No credentials of their own at any step.

Three weaknesses chained: public S3 exposed the credentials. Open SSH got them in. AdministratorAccess IAM role meant they didn't need their own AWS credentials to exfiltrate.

Checkov flagged the bucket. It flagged the IAM role. Inspector flagged the SSH port. But no tool said: these three connect into a complete attack path. That's the gap.

Wiz builds a graph of your entire environment — every resource, every permission, every network path — and identifies which combinations of findings create real attack paths. Instead of 40 isolated findings, you get three prioritised attack paths. Your team fixes what actually matters, not what shows up first in a list.

You can keep adding tools — Semgrep, KICS, others. Each adds signal but also another dashboard. Wiz sits above all of them and gives you one view with the context to act.

---

## Slide 10 — Challenges & What I Learned

Building this wasn't without friction — and the friction is where the real learning happened.

Four challenges:

First — cross-platform Docker builds. My Mac runs ARM64, EKS runs AMD64. The container built fine locally and crashed immediately on Kubernetes. Fixed with docker buildx and --platform linux/amd64. Container portability isn't automatic — you have to be explicit about target architecture.

Second — the AWS Load Balancer Controller couldn't discover my subnets because they were missing required Kubernetes cluster tags. Added the tags via Terraform and pinned subnet IDs in the ingress. The ALB controller has very specific expectations — and when they're not met, the error messages aren't obvious.

Third — Streamlit uses WebSockets. Behind an ALB without sticky sessions, every page refresh broke the connection. Fixed with sticky session annotations on the ingress. Not all HTTP traffic behaves the same.

Fourth — Terraform state. Moving to S3 made the pipeline work — but the state file contains credentials in plain text, and I'd put it in a public bucket. That mistake became the strongest demo moment in the presentation.

Key takeaway: the most dangerous finding wasn't open SSH or outdated software. It came from combining two unrelated decisions — a public S3 bucket and Terraform remote state. Neither alarming on its own. Together they create a complete attack path. That's the insight that made the Wiz value proposition click for me — it's not about finding more things, it's about connecting what you already know.

---

## Slide 11 — What I'd Do Differently

Four changes in priority order:

First — AWS Secrets Manager with the External Secrets Operator. Credentials out of Kubernetes Secrets and Terraform state. Stored encrypted in AWS, rotated automatically, synced into Kubernetes at runtime. Nothing sits in a file anywhere.

Second — IMDSv2 on EC2 and immutable ECR image tags. IMDSv2 adds a required session token to metadata requests — blocks the most common way attackers steal instance credentials without being on the machine. Immutable ECR tags mean a scanned image can't be overwritten with a different one after the fact.

[IF ASKED about IMDSv2]
The classic attack is SSRF — the app is tricked into calling 169.254.169.254 on behalf of an attacker. IMDSv2 requires a PUT request first to get a session token. SSRF can't do that. One line in Terraform: http_tokens = required.

Third — OIDC federation instead of static AWS credentials in GitHub Secrets. GitHub requests a short-lived token from AWS per pipeline run. Nothing stored, nothing to rotate, nothing to leak.

[IF ASKED about OIDC]
GitHub and AWS trust each other via OpenID Connect. GitHub says "I am this repo, running this workflow" — AWS issues a temporary credential scoped to one IAM role. Expires in minutes.

Fourth — move MongoDB to a private subnet with no public IP, and add a VPC S3 endpoint. Moving to a private subnet removes the public IP entirely — the machine can't be reached from outside AWS at all, even with port 22 open. But the daily backup script uses the AWS CLI to copy backups to S3, and a private subnet has no internet gateway route — that call would break. A VPC S3 endpoint solves this by creating a private route directly from the VPC to S3, without traffic ever leaving the AWS network.

[IF ASKED about VPC endpoints]
Our pod-to-MongoDB traffic already stays inside the VPC via private IPs — that's working correctly. The VPC endpoint is specifically for EC2-to-S3 traffic. Without it, the backup script would go out through the internet to reach S3. The endpoint keeps that private and is a prerequisite for the private subnet move to work end-to-end.

---

## Slide 12 — Close / Q&A

That brings me to the end. To summarise: I built a real working cloud-native application, deliberately introduced the required security weaknesses, built a DevSecOps pipeline that surfaces them automatically, and implemented AWS-native controls for detection and audit. And I showed how the fragmentation of findings across those tools is exactly the problem Wiz was built to solve.

I'd love to open it up for questions — and I'd genuinely appreciate your feedback. Is there anything you'd like me to go deeper on, or anything you'd have done differently?

[After Q&A]

Thank you all. I'll follow up by email with the GitHub repository link. Looking forward to the next steps.
