data "aws_caller_identity" "current" {}

# Rule 1: SSH must not be open to the internet
# Will immediately flag movie-buddy-ec2-sg as NON_COMPLIANT (intentional weakness)
resource "aws_config_config_rule" "restricted_ssh" {
  name = "${var.project_name}-restricted-ssh"

  source {
    owner             = "AWS"
    source_identifier = "INCOMING_SSH_DISABLED"
  }
}

# Rule 2: S3 buckets must not allow public read access
# Will immediately flag movie-buddy-tfstate bucket as NON_COMPLIANT (intentional weakness)
resource "aws_config_config_rule" "s3_public_read_prohibited" {
  name = "${var.project_name}-s3-public-read-prohibited"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_PUBLIC_READ_PROHIBITED"
  }
}
