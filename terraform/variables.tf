variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-central-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "public_key" {
  description = "SSH public key — paste the contents of your .pem.pub or id_rsa.pub"
  type        = string
  sensitive   = true
}

variable "allowed_cidrs" {
  description = "CIDRs allowed to reach the API (port 8000) and Streamlit (port 8501)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "repo_url" {
  description = "Git repo to clone onto the instance"
  type        = string
  default     = "https://github.com/oleksii-androsov/ai-learning.git"
}
