data "aws_ami" "ubuntu_20_04" {
  most_recent = true
  owners      = ["099720109477"] # Canonical (Ubuntu's official AWS account)

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}

# Intentional weakness: overly permissive IAM role (can create VMs)
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_admin" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# Intentional weakness: SSH open to the entire internet
resource "aws_security_group" "ec2_sg" {
  name   = "${var.project_name}-ec2-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    description = "SSH from anywhere (intentional weakness)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "MongoDB from VPC only"
    from_port   = 27017
    to_port     = 27017
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-ec2-sg" }
}

resource "aws_key_pair" "deployer" {
  key_name   = "${var.project_name}-key"
  public_key = file("${path.module}/../wiz-exercise-key.pub")
}

resource "aws_instance" "mongodb" {
  ami                    = data.aws_ami.ubuntu_20_04.id
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = aws_key_pair.deployer.key_name

  user_data = <<-EOF
    #!/bin/bash
    set -e

    # Install MongoDB 4.4 (intentionally outdated — 1+ year old)
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add -
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" \
      > /etc/apt/sources.list.d/mongodb-org-4.4.list
    apt-get update -y
    apt-get install -y mongodb-org=4.4.29 mongodb-org-server=4.4.29

    # Enable auth and bind to all interfaces so EKS can connect
    cat > /etc/mongod.conf <<MONGOCFG
    storage:
      dbPath: /var/lib/mongodb
    systemLog:
      destination: file
      path: /var/log/mongodb/mongod.log
      logAppend: true
    net:
      port: 27017
      bindIp: 0.0.0.0
    security:
      authorization: enabled
    MONGOCFG

    systemctl enable mongod
    systemctl start mongod

    # Wait for MongoDB to start then create admin user
    sleep 10
    mongo --eval "
      db = db.getSiblingDB('admin');
      db.createUser({
        user: '${var.mongodb_username}',
        pwd: '${var.mongodb_password}',
        roles: [{ role: 'root', db: 'admin' }]
      });
    "

    # Install AWS CLI for backups
    apt-get install -y awscli

    # Daily backup script to S3
    cat > /usr/local/bin/backup-mongodb.sh <<BACKUP
    #!/bin/bash
    DATE=\$(date +%Y%m%d-%H%M%S)
    mongodump --uri="mongodb://${var.mongodb_username}:${var.mongodb_password}@localhost:27017" \
      --out /tmp/backup-\$DATE
    tar -czf /tmp/backup-\$DATE.tar.gz -C /tmp backup-\$DATE
    aws s3 cp /tmp/backup-\$DATE.tar.gz s3://${var.project_name}-db-backups/\$DATE.tar.gz
    rm -rf /tmp/backup-\$DATE /tmp/backup-\$DATE.tar.gz
    BACKUP

    chmod +x /usr/local/bin/backup-mongodb.sh

    # Run backup daily at 2am
    echo "0 2 * * * root /usr/local/bin/backup-mongodb.sh" > /etc/cron.d/mongodb-backup
  EOF

  tags = { Name = "${var.project_name}-mongodb" }
}
