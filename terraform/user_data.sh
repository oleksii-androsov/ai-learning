#!/bin/bash
set -e
exec > /var/log/user_data.log 2>&1

apt-get update -y
apt-get install -y python3 python3-venv python3-pip git

git clone ${repo_url} /home/ubuntu/ai-learning
chown -R ubuntu:ubuntu /home/ubuntu/ai-learning

cd /home/ubuntu/ai-learning
sudo -u ubuntu python3 -m venv .venv
sudo -u ubuntu .venv/bin/pip install --quiet -r week04-rag-api/requirements.txt
sudo -u ubuntu .venv/bin/pip install --quiet ddtrace

# Fix directory permissions so Datadog agent can read logs
chmod o+x /home/ubuntu

cat > /etc/systemd/system/rag-api.service <<'EOF'
[Unit]
Description=RAG API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-learning
EnvironmentFile=/home/ubuntu/ai-learning/.env
ExecStart=/home/ubuntu/ai-learning/.venv/bin/ddtrace-run /home/ubuntu/ai-learning/.venv/bin/uvicorn week04-rag-api.api:app --host 0.0.0.0 --port 8000
Restart=on-failure
StandardOutput=append:/home/ubuntu/ai-learning/server.log
StandardError=append:/home/ubuntu/ai-learning/server.log

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/rag-streamlit.service <<'EOF'
[Unit]
Description=RAG Streamlit
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-learning
EnvironmentFile=/home/ubuntu/ai-learning/.env
ExecStart=/home/ubuntu/ai-learning/.venv/bin/streamlit run /home/ubuntu/ai-learning/week04-rag-api/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Restart=on-failure
StandardOutput=append:/home/ubuntu/ai-learning/streamlit.log
StandardError=append:/home/ubuntu/ai-learning/streamlit.log

[Install]
WantedBy=multi-user.target
EOF

# Allow ubuntu to restart services without a password (needed for CI/CD)
echo "ubuntu ALL=(ALL) NOPASSWD: /bin/systemctl restart rag-api rag-streamlit, /bin/systemctl start rag-api rag-streamlit, /bin/systemctl stop rag-api rag-streamlit" \
  > /etc/sudoers.d/rag-services

systemctl daemon-reload
systemctl enable rag-api rag-streamlit
# Services start once the user copies .env onto the instance
